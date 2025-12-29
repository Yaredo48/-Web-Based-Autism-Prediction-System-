import pandas as pd
import numpy as np
from pathlib import Path
import logging # It helps you see what your API is doing


class DataCollector:
    def __init__(self, data_dir: Path = Path("data/")):
        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        #logging.basicConfig(level=logging.INFO)
        
        def load_autism_dataset(self, source: str = "uci"):
            """
            Load the autism dataset from a specified source.
            
            Parameters:
            source (str): The source from which to load the dataset. Default is "uci".
            
            Returns:
            pd.DataFrame: The loaded autism dataset.
            """
            if source == "uci":
                # UCI Autism Screening Adult Dataset
                url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00426/Autism-Adult-Data%20Plus%20Description%20File.zip"
                return self.download_and_load(url)
            elif source == "kaggle":
                # Kaggle Autism Screening Dataset
                return self._load_from_kaggle()
            else:
                raise ValueError(f"Unknown source: {source}")
            
            
                
                '''url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00451/dataR2.csv"
                self.logger.info(f"Loading autism dataset from {url}")
                df = pd.read_csv(url)
                self.logger.info("Autism dataset loaded successfully")
                return df
            else:
                self.logger.error(f"Source {source} not recognized.")
                raise ValueError(f"Source {source} not recognized.")'''
        
        def download_and_load(self, url):
            try:
                # Download and extract the dataset
                import requests
                import zipfile
                import io
                
                self.logger.info(f"Downloading dataset from {url}")
                response = requests.get(url)
                zip_file= zipfile.ZipFile(io.BytesIO(response.content))
                zip_file.extractall(self.data_dir)
                self.logger.info("Dataset downloaded and extracted successfully")
                
                # find csv files
                
                csv_files = list(self.data_dir.glob("*.csv"))
                if csv_files:
                    df = pd.read_csv(csv_files[0])
                    self.logger.info(f"Loaded data shape: {df.shape}")
                    return df
            except exception as e:
                self.logger.error(f"Failed to download or load dataset: {e}")
                raise e
            # falback to sample data generation
            self.logger.info("Generating sample autism dataset")
            return self._generate_sample_data()
        
        
        def _generate_sample_data(self):
            np.random.seed(42)
            n_samples=1000
            
            data = {
            'age': np.random.randint(2, 65, n_samples),
            'gender': np.random.choice(['m', 'f'], n_samples),
            'ethnicity': np.random.choice(['White', 'Asian', 'Middle Eastern', 'Others'], n_samples),
            'jaundice': np.random.choice([True, False], n_samples),
            'autism_history': np.random.choice([True, False], n_samples),
            'score_A1': np.random.randint(0, 2, n_samples),
            'score_A2': np.random.randint(0, 2, n_samples),
            'score_A3': np.random.randint(0, 2, n_samples),
            'score_A4': np.random.randint(0, 2, n_samples),
            'score_A5': np.random.randint(0, 2, n_samples),
            'score_A6': np.random.randint(0, 2, n_samples),
            'score_A7': np.random.randint(0, 2, n_samples),
            'score_A8': np.random.randint(0, 2, n_samples),
            'score_A9': np.random.randint(0, 2, n_samples),
            'score_A10': np.random.randint(0, 2, n_samples),
            'result': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
            }
            
            df=pd.DataFrame(data)
            self.logger.info(f"Sample autism dataset generated with shape: {df.shape}")
            df.to_csv(self.data_dir / "autism_screening_sample.csv", index=False)
            return df



        def explore_data(self, df):
             
                # Perform EDA
                eda_results = {
                'shape': df.shape,
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'basic_stats': df.describe().to_dict(),
                'correlation': df.select_dtypes(include=[np.number]).corr().to_dict()
        }
                return eda_results
            # OR 
            
                """
                Explore the dataset by providing basic statistics and information.
                
                Parameters:
                df (pd.DataFrame): The dataset to explore.
                
                self.logger.info("Exploring dataset...")
                self.logger.info(f"Dataset shape: {df.shape}")
                self.logger.info(f"Dataset columns: {df.columns.tolist()}")
                self.logger.info("Dataset info:")
                self.logger.info(df.info())
                self.logger.info("Dataset description:")
                self.logger.info(df.describe(include='all')) """
        
                                
        
        