import os
import torch
import uproot
import awkward as ak
import numpy as np
from tqdm import tqdm
import urllib.parse
import tarfile
import zipfile
import shutil
import requests
import hashlib

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import knn_graph
from torch_geometric.utils import dense_to_sparse


def extract_archive(file_path, path='.', archive_format='auto'):
    """Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats."""
    if archive_format is None:
        return False
    if archive_format == 'auto':
        archive_format = ['tar', 'zip']
    if isinstance(archive_format, str):
        archive_format = [archive_format]

    for archive_type in archive_format:
        if archive_type == 'tar':
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        if archive_type == 'zip':
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile

        if is_match_fn(file_path):
            with open_fn(file_path) as archive:
                try:
                    archive.extractall(path)
                except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                    if os.path.exists(path):
                        if os.path.isfile(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
                    raise
            return True
    return False


def _download(url, fpath):
    """Download file from URL with retry logic."""
    import time
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(fpath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return
            
        except (requests.exceptions.RequestException, requests.exceptions.ChunkedEncodingError) as e:
            if attempt < max_retries - 1:
                print(f"Download attempt {attempt + 1} failed: {e}")
                print(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                raise


def validate_file(fpath, file_hash, algorithm='md5'):
    """Validate file hash."""
    if algorithm == 'md5':
        hasher = hashlib.md5()
    elif algorithm == 'sha256':
        hasher = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    
    return hasher.hexdigest() == file_hash


def get_file(origin=None, fname=None, file_hash=None, datadir='datasets',
             hash_algorithm='md5', extract=False, force_download=False,
             archive_format='auto'):
    """Downloads a file from a URL if it not already in the cache."""
    if origin is None:
        raise ValueError('Please specify the "origin" argument (URL of the file to download).')

    os.makedirs(datadir, exist_ok=True)

    if not fname:
        fname = os.path.basename(urllib.parse.urlsplit(origin).path)
        if not fname:
            raise ValueError(f"Can't parse the file name from the origin provided: '{origin}'.")

    fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath) and not force_download:
        print(f'A local file already found at {fpath}')
        if file_hash is not None:
            if validate_file(fpath, file_hash, algorithm=hash_algorithm):
                print('Local file hash matches, no need to download.')
            else:
                print('Local file hash mismatch, re-downloading...')
                download = True
        else:
            print('No hash provided, using existing file.')
    else:
        download = True

    if download:
        print(f'Downloading data from {origin} to {fpath}')
        try:
            _download(origin, fpath)
        except Exception as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise Exception(f'Download failed from {origin}: {str(e)}')  # Fixed this line

        if file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                if os.path.exists(fpath):
                    os.remove(fpath)
                raise RuntimeError(f'Checksum does not match for file {fpath}')

    if extract:
        extract_archive(fpath, datadir, archive_format)

    return fpath, download


# -------------------------------
# Step 1: Load ROOT -> Awkward
# -------------------------------
def fileToAwk(path_or_url):
    """Read ROOT file from local path or remote URL."""
    try:
        file = uproot.open(path_or_url)
        tree = file['tree']
        awk = tree.arrays(tree.keys())
        return awk
    except Exception as e:
        print(f"Error reading {path_or_url}: {e}")
        return None


# -------------------------------
# Step 2: Awkward -> Point cloud
# -------------------------------
input_features = [
    "part_px", "part_py", "part_pz", "part_energy",
    "part_deta", "part_dphi", "part_d0val", "part_d0err",
    "part_dzval", "part_dzerr", "part_isChargedHadron",
    "part_isNeutralHadron", "part_isPhoton", "part_isElectron", "part_isMuon"
]

def awkToPointCloud(awkDict, input_features, max_jets=1000):
    """Convert awkward arrays to point clouds, limiting number of jets for testing."""
    featureVector = []
    num_jets = min(len(awkDict), max_jets)  # Limit for faster processing
    
    for jet in tqdm(range(num_jets), desc="Converting jets to point clouds"):
        currJet = awkDict[jet][input_features]
        
        # Check if jet has particles
        if len(currJet["part_px"]) == 0:
            continue
            
        pT = np.sqrt(currJet["part_px"]**2 + currJet["part_py"]**2)

        currJet_array = np.column_stack((
            np.array(currJet["part_px"]),
            np.array(currJet["part_py"]),
            np.array(currJet["part_pz"]),
            np.array(currJet["part_energy"]),
            pT,
            np.array(currJet["part_deta"]),
            np.array(currJet["part_dphi"]),
            np.array(currJet["part_d0val"]),
            np.array(currJet["part_d0err"]),
            np.array(currJet["part_dzval"]),
            np.array(currJet["part_dzerr"]),
            np.array(currJet["part_isChargedHadron"]),
            np.array(currJet["part_isNeutralHadron"]),
            np.array(currJet["part_isPhoton"]),
            np.array(currJet["part_isElectron"]),
            np.array(currJet["part_isMuon"])
        ))

        featureVector.append(currJet_array)
    
    return featureVector


# -------------------------------
# Step 3A: Build KNN Graph
# -------------------------------
def pointCloudToKNNGraph(pointCloud, k, label):
    """Convert point cloud to KNN graph."""
    if len(pointCloud) == 0:
        return None
        
    x = torch.tensor(pointCloud, dtype=torch.float32)
    
    # Use spatial coordinates for KNN (px, py, pz)
    edge_index = knn_graph(x[:, :3], k=min(k, x.size(0)-1), loop=False)
    
    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


# -------------------------------
# Step 3B: Build Fully Connected Graph
# -------------------------------
def pointCloudToFullyConnectedGraph(pointCloud, label):
    """Convert point cloud to fully connected graph."""
    if len(pointCloud) == 0:
        return None
        
    x = torch.tensor(pointCloud, dtype=torch.float32)
    N = x.size(0)
    
    # Create fully connected adjacency matrix (excluding self-loops)
    adj = torch.ones((N, N)) - torch.eye(N)
    edge_index, _ = dense_to_sparse(adj)
    
    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


# -------------------------------
# Step 4: Dataset Class
# -------------------------------
class JetDataset(InMemoryDataset):
    # Always use the small validation dataset (5M)
    urls = {
        "val": [
            ("https://zenodo.org/record/6619768/files/JetClass_Pythia_val_5M.tar",
             "7235ccb577ed85023ea3ab4d5e6160cf"),
        ]
    }

    def __init__(self, root, jet_map, graph_type="knn", k=3, 
                 transform=None, pre_transform=None, use_remote=False, 
                 skip_download=False, max_jets_per_file=500):
        """
        Args:
            root: Root directory for dataset
            jet_map: dict {class_label: [list of ROOT file paths]}
            graph_type: "knn" or "fully"
            k: Number of nearest neighbors for KNN graph
            use_remote: If True, use remote URLs directly
            skip_download: If True, skip download and use local files
            max_jets_per_file: Maximum number of jets to process per file
        """
        self.jet_map = jet_map
        self.graph_type = graph_type
        self.k = k
        self.use_remote = use_remote
        self.skip_download = skip_download
        self.max_jets_per_file = max_jets_per_file
        
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        """Return expected raw file names."""
        files = []
        for label, fileList in self.jet_map.items():
            files.extend(fileList)
        return files

    @property
    def processed_file_names(self):
        """Return processed file names."""
        return [f"data_{self.graph_type}_k{self.k}_small.pt"]

    def download(self):
        """Download the small validation dataset."""
        if self.skip_download or self.use_remote:
            print("Skipping download - using existing/remote files")
            return
            
        datadir = self.raw_dir
        url, md5 = self.urls["val"][0]  # Always use validation set
        
        print(f"Downloading small validation dataset (5M)...")
        fpath, downloaded = get_file(url, datadir=datadir, file_hash=md5, 
                                   force_download=False)
        if downloaded:
            print("Extracting archive...")
            extract_archive(fpath, path=datadir)

    def process(self):
        """Process ROOT files into graph data."""
        print("Processing ROOT files into graphs...")
        data_list = []
        
        for label, fileList in self.jet_map.items():
            print(f"Processing label {label} with {len(fileList)} files")
            
            for filepath in fileList:
                if self.use_remote:
                    full_path = filepath  # Assume filepath is already a URL
                else:
                    full_path = os.path.join(self.raw_dir, filepath)
                
                print(f"Loading: {full_path}")
                awk = fileToAwk(full_path)
                
                if awk is None:
                    print(f"Failed to load {full_path}, skipping...")
                    continue
                
                # Convert to point clouds with limited number of jets
                clouds = awkToPointCloud(awk, input_features, 
                                       max_jets=self.max_jets_per_file)
                
                print(f"Converting {len(clouds)} jets to graphs...")
                for pc in tqdm(clouds, desc=f"Creating graphs for label {label}"):
                    if len(pc) == 0:
                        continue
                        
                    try:
                        if self.graph_type == "knn":
                            graph = pointCloudToKNNGraph(pc, self.k, label)
                        elif self.graph_type == "fully":
                            graph = pointCloudToFullyConnectedGraph(pc, label)
                        else:
                            raise ValueError("graph_type must be 'knn' or 'fully'")
                        
                        if graph is not None:
                            data_list.append(graph)
                            
                    except Exception as e:
                        print(f"Error creating graph: {e}")
                        continue

        print(f"Created {len(data_list)} graphs total")
        
        if len(data_list) == 0:
            raise ValueError("No valid graphs were created!")
        
        # Collate all graphs and save
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Saved processed data to {self.processed_paths[0]}")
