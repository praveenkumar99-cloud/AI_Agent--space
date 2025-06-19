import os
import json
import time
import glob
import numpy as np
import pandas as pd
import requests
import logging
import re
import os
import tqdm
import datetime
import concurrent.futures
from multiprocessing import Pool
from itertools import islice
from pathlib import Path
from datetime import date
from dataclasses import dataclass
from google.cloud import bigquery
from jsonschema import validate, ValidationError
from typing import List, Dict, Optional, Generator
from src.main.jobs.config.consts import GasConfig
from src.main.jobs.storage.Bucket import CachedBucketService
from src.main.jobs.config.category_config import Config
from src.main.jobs.productrelevancy.product_info_bq_client import get_products_from_bqm
from src.main.jobs.productrelevancy.abstract_batch_manager import AbstractBatchJobMonitor,AbstractBatchRequestBuilder,AbstractBatchJobManager
from src.main.jobs.productrelevancy.prompt_generator_applicated import PromptGenerator as ApplicatedPromptGenerator
from src.main.jobs.productrelevancy.prompt_generator_non_applicated import PromptGenerator as NonApplicatedPromptGenerator

log = logging.getLogger(__name__)
gas_config = GasConfig()
processed_log = "processed_chunks.jsonl"
failed_log = "failed_chunks.jsonl"

def PromptGenerator():
   df = get_products_from_bqm("temp")
   is_applicated = df.get("is_applicated")
   if isinstance(is_applicated, pd.Series):
    is_applicated = is_applicated.iloc[0]
   if isinstance(is_applicated, str):
    is_applicated = is_applicated.lower() == "true"
   elif isinstance(is_applicated, (bool, int)):
    is_applicated = bool(is_applicated)
   else:
    is_applicated = False  
   if is_applicated:
    prompt_generator = ApplicatedPromptGenerator(Config.MODEL_ID)
   else:
    prompt_generator = NonApplicatedPromptGenerator(Config.MODEL_ID)
   return prompt_generator

def load_processed_chunks():
    if Path(processed_log).exists():
        try:
            with open(processed_log, 'r') as f:
                return set(map(str,json.load(f)))
        except Exception as e:
            log.info("failed to load processed chunks: {e}")
            return set()
    return set()

def save_processed_chunk(file_path):
    processed= load_processed_chunks()
    processed.add(str(file_path))
    with open(processed_log, 'w') as f:
        json.dump(sorted(list(processed)),f)

def save_failed_chunk(part_file_path: str):
   failed_path = Path("failed_chunks.jsonl")
   failed = set()
   if failed_path.exists():
       with open(failed_path, "r") as f:
           for line in f:
               try:
                   failed.add(json.loads(line.strip())["chunk"])
               except json.JSONDecodeError as e:
                   log.error(f"Skipping invalid JSON line: {line.strip()} ({e})")
   failed.add(str(part_file_path))
   with open(failed_path, "w") as f:
       for chunk in failed:
           f.write(json.dumps({"chunk": chunk}) + "\n")
   log.info(f"Saved failed chunk: {part_file_path}")

def clear_processed_chunks():
    if(Path(processed_log).exists()):
        Path(processed_log).unlink()
        log.info("cleared processed_chunks - ready for new batch to run")

def split_jsonl_file(file_path: str, max_lines: int = 2000) -> list:
   """
   Splits a large JSONL file into smaller parts.
   Returns the list of file paths.
   """
   split_file_paths = []
   with open(file_path, 'r') as src:
       for i, chunk in enumerate(iter(lambda: list(islice(src, max_lines)), [])):
           part_file_path = f"{file_path.replace('.jsonl', '')}_part_{i}.jsonl"
           with open(part_file_path, 'w') as part_file:
               part_file.writelines(chunk)
           split_file_paths.append(part_file_path)
   return split_file_paths
def cleanup_old_input_files(base_file_name):
   base_name = base_file_name.replace('.jsonl', '')
   full_file = base_file_name
   chunk_pattern = f"{base_name}_part_*.jsonl"
   if os.path.exists(full_file):
       os.remove(full_file)
       log.info(f"Deleted old batch request file: {full_file}")
   for chunk_file in glob.glob(chunk_pattern):
       try:
           os.remove(chunk_file)
           log.info(f"Deleted old chunk file: {chunk_file}")
       except Exception as e:
           log.warning(f"Could not delete {chunk_file}: {e}")
   if os.path.exists("processed_chunks.json"):
       os.remove("processed_chunks.json")
       log.info("️Deleted old processed_chunks.json")

@dataclass
class MonitorConfig:
   access_token: str
   bucket_service_output: CachedBucketService
   bq_client: bigquery.Client
   bq_table_id: str

class GasBatchJobMonitor(AbstractBatchJobMonitor):
   def __init__(self, batch_id: str, monitor_config: MonitorConfig):
       self.batch_id = batch_id
       self._access_token = monitor_config.access_token
       self.bucket_service_output = monitor_config.bucket_service_output
       self.bq_client = monitor_config.bq_client
       self.bq_table_id = monitor_config.bq_table_id
       self.MAX_RETRIES = 2
       self.predictions_file_name = (
           f"Batch_predictions_output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
       )
   def _download_and_save_batch_results(self, output_file_id: str):
       
       log.info(f"Downloading batch results, file_id: {output_file_id}")
       try:
           response = requests.get(
               f"{gas_config.apigee_base_url}/genaisp/gas/files/{output_file_id}/content",
               headers={"Authorization": f"Bearer {self._access_token}"},
           )
           response.raise_for_status()
           file_content = response.content.decode("utf-8")
       except requests.exceptions.RequestException as e:
           log.error(f"Failed to download batch results: {e}", exc_info=True)
           return None
       if not file_content.strip():
           log.error("Downloaded file is empty.")
           return None
       log.debug("First 500 characters of downloaded content:")
       log.debug(repr(file_content[:500]))
       log.info("Writing predictions locally...")
       with open(self.predictions_file_name, "w") as f:
           for line_num, json_string in enumerate(file_content.splitlines(), 1):
               if json_string.strip():
                   try:
                       json_data = json.loads(json_string)
                       f.write(json.dumps(json_data) + "\n")
                   except json.JSONDecodeError as e:
                       log.error(
                           f"[Line {line_num}] Failed to decode JSON: {e}, Content: {json_string}",
                           exc_info=True,
                       )
               else:
                   log.warning(f"[Line {line_num}] Skipped empty or whitespaced only line")
       log.info(f"File saved to: {self.predictions_file_name}")
       log.debug(f"Full downloaded file content:\n{file_content}")
       self.bucket_service_output.add(self.predictions_file_name)
       return self.predictions_file_name
   def _define_product_schema(self):  
       return {
           "type": "array",
           "items": {
               "type": "object",
               "properties": {
                   "productId": {"type": "string"},
                   "title": {"type": "string"},
                   "position": {"type": "integer"},
                   "relevanceScore": {"type": "integer", "minimum": 0, "maximum": 4},
                   "explanation": {"type": "string"},
                   "env": {"type": "string"}
               },
               "required": ["productId", "title", "position", "relevanceScore", "explanation", "env"],
           },
       }
   def _upload_to_bigquery(self, results: List[dict], df: pd.DataFrame):
       log.info("Uploading results to BigQuery...")
       log.debug(f"Incoming Dataframe columns: {df.columns.tolist()}")
       client = self.bq_client
       table_id = self.bq_table_id
       rows_to_insert: List[Dict] = []
       product_schema = self._define_product_schema()
       for result in results:
           custom_id_str = result.get("custom_id", "-1")
           try:
               original_row_id = int(custom_id_str.split("_")[0])
           except ValueError:
               log.error(
                   f"Could not extract original row ID from custom_id: {custom_id_str}",
                   exc_info=True,
               )
               continue
           if 0 <= original_row_id < len(df):
               search_term = df.iloc[original_row_id]["searchQuery"]
               channel = df.iloc[original_row_id]["channel"]
               Is_applicated = df.iloc[original_row_id]["is_applicated"]
               products = df.iloc[original_row_id]["products"]
               application_title = df.iloc[original_row_id]["application_title"]
               if isinstance(products, np.ndarray):
                   products = products.tolist()
               response_content = result["response"]["body"]["choices"][0]["message"]["content"]
               logging.debug(f"Original response content for custom_id {custom_id_str}: {repr(response_content)}")
               response_content = response_content.strip()
               response_content = re.sub(r"```json\n","",response_content)
               response_content = re.sub(r"```","",response_content)
               response_content = re.sub(r";",",",response_content)  
               try:
                validation_data = json.loads(response_content)
               except json.JSONDecodeError as e:
                   log.error(f"failed to parse JSON for custom_id {custom_id_str}: {e}. content: {response_content}")
                   with open("failed_responses.log", "a") as f:
                       f.write(f"custom_id: {custom_id_str}\n{response_content}\n---\n")
                       continue

               if validation_data is None:
                   logging.error(
                       f"Failed to parse JSON for custom_id {custom_id_str}. Content: {response_content}",
                   )
                   with open("failed_responses.log", "a") as f:
                       f.write(f"custom_id:{custom_id_str}\n{response_content}\n---\n")
                   continue
               try:
                   validate(instance=validation_data, schema=product_schema)
               except ValidationError as ve:
                   logging.error(
                       f"JSON schema validation failed for custom_id {custom_id_str}: {ve}. Content: {response_content}",
                   )
                   with open("failed_responses.log", "a") as f:
                       f.write(f"custom_id:{custom_id_str}\nSchema validation error: {ve}\nContent:{response_content}\n---\n")
                   continue
               for product_info in validation_data:
                   if (
                       "position" in product_info
                       and "productId" in product_info
                       and "relevanceScore" in product_info
                       and "explanation" in product_info
                       and "env" in product_info
                       
                   ):
                       rows_to_insert.append(
                           {
                               "Search_term": search_term,
                               "Product_title": application_title,
                               "position": product_info["position"],
                               "UAT_or_PROD": product_info["env"],
                               "Product_id": product_info["productId"],
                               "Selling_Channel": channel,
                               "is_applicated":bool(Is_applicated),
                               "Load_date": datetime.datetime.now().strftime('%Y-%m-%d'),
                               "relevance_score": int(product_info["relevanceScore"]),
                               "explanation": product_info["explanation"].replace("\\", "")
   
                           }
                       )
                   else:
                       log.error(f"Missing expected keys in product_info: {product_info}")
           else:
               log.error(
                   f"Original row ID {original_row_id} is out of bounds for DataFrame of length {len(df)}"
               )
       if not rows_to_insert:
           log.warning("No rows to insert into BigQuery.")
           return
       try:
           errors = client.insert_rows_json(table_id, rows_to_insert)
           if errors:
               log.error(f"Encountered errors while inserting rows into BigQuery: {errors}")
           else:
               log.info("Rows successfully inserted into BigQuery")
       except Exception as e:
           log.exception(f"Error inserting rows into BigQuery", exc_info=True)
   def cancel_batch_job(self):
       """Force stop a GAS  batch job if supported by the API"""
       token_response = requests.post(
           f"{gas_config.apigee_base_url}/oauthv2/token",
           data={
               "grant_type": "client_credentials",
               "client_id": gas_config.client_id,
               "client_secret": gas_config.client_secret,
           },
       )
       if token_response.status_code != 200:
           log.error(f"Failed to get access token: {token_response.text}")
           return
       access_token = token_response.json().get("access_token")
       if not access_token:
           log.error("Access token not found in token response")
           return
       cancel_url = f"{gas_config.apigee_base_url}/genaisp/gas/batches/{self.batch_id}"
       try:
           response = requests.delete(
               cancel_url, headers={"Authorization": f"bearer {self._access_token}"}
           )
           response.raise_for_status()
           if response.status_code in (200, 204):
               log.info(f"Batch job {self.batch_id} successfully cancelled.")
           else:
               log.error(
                   f"Failed to cancel batch job {self.batch_id}. status: {response.status_code}, response: {response.text}"
               )
       except requests.exceptions.RequestException as e:
           log.error(f"Error cancelling batch job: {e}", exc_info=True)
   
   def monitor(self, refresh_time: int = 500, max_monitor_duration: int = 3000, df: pd.DataFrame = None) -> str:
       start_time = time.time()
       status_not_updated_num = 0
       log.info(f"Starting job monitoring for batch_id: {self.batch_id}...")
       current_job_status: Optional[str] = None
       while current_job_status not in ("expired", "failed", "completed"):
           time.sleep(refresh_time)
           try:
               batch_retrieved = requests.get(
                   f"{gas_config.apigee_base_url}/genaisp/gas/batches/{self.batch_id}",
                   headers={"Authorization": f"Bearer {self._access_token}"},
               )
               batch_retrieved.raise_for_status()
               batch_data = batch_retrieved.json()
               current_job_status = batch_data["status"]
               req_cnt_dict = batch_data.get("request_counts", {})
               log.info(
                   "PROGRESS UPDATE -> "
                   f"TOTAL: {req_cnt_dict.get('total')}, "
                   f"COMPLETED: {req_cnt_dict.get('completed')}, "
                   f"FAILED {req_cnt_dict.get('failed')}"
               )
               status_not_updated_num = 0
           except requests.exceptions.RequestException as e:
               status_not_updated_num += 1
               log.warning(
                   f"Status not updated! Will retry... Retry num: {status_not_updated_num}, Error: {e}",
                   exc_info=True,
               )
           except json.JSONDecodeError as e:
               status_not_updated_num += 1
               log.warning(
                   f"Failed to decode batch status response: {e}, Content: {batch_retrieved.text if batch_retrieved else ''}",
                   exc_info=True,
               )
           except KeyError as e:
               status_not_updated_num += 1
               log.warning(
                   f"Missing key in batch status response: {e}, Content: {batch_retrieved.text if batch_retrieved else ''}",
                   exc_info=True,
               )
           if status_not_updated_num == self.MAX_RETRIES:
               log.error(f"Retry limit for status update has reached {self.MAX_RETRIES}. Aborting...")
               return None
           elapsed_time = time.time() - start_time
           log.info(
               f"Current status: {str(current_job_status).upper()}. Job running.. Elapsed time: {elapsed_time:.2f} seconds"
           )
           if elapsed_time > max_monitor_duration:
               log.warning(
                   f"Monitoring duration exceeded max_monitor_duration ({max_monitor_duration} seconds). Cancelling job."
               )
               self.cancel_batch_job()
               return None
       if current_job_status != "completed":
           log.error(f"Batch job status finished with status: {str(current_job_status).upper()}")
           return None
       try:
           output_file_id = batch_data["output_file_id"]
           log.info(f"Received output_file_id: {output_file_id}")
           log.debug(f"Batch status response: {batch_data}")
           gcs_path_batch = self._download_and_save_batch_results(output_file_id)
           if gcs_path_batch:
               with open(gcs_path_batch, "r") as f:
                   results = [json.loads(line) for line in f if line.strip()]
               self._upload_to_bigquery(results, df)
               os.remove(self.predictions_file_name)
               return gcs_path_batch
           else:
               log.error("Failed to download or save batch results. Monitor failed.")
               return None
       except KeyError as e:
           log.error(f"Missing key in batch response: {e}, Response: {batch_data}", exc_info=True)
           return None
       except Exception as e:
           log.exception(f"Unexpected error during monitoring: {e}", exc_info=True)
           return None

class GasBatchRequestBuilder(AbstractBatchRequestBuilder):
   def __init__(
       self,
       df: pd.DataFrame,
       prompt_generator: PromptGenerator,
       config: Config,
       max_prompt_tokens: int = 3800,
       products_per_prompt: int = 3,
   ):
       self.df = df
       self.prompt_generator = prompt_generator
       self.config = config
       self.max_prompt_tokens = max_prompt_tokens
       self.products_per_prompt = products_per_prompt
   def build(self, file_name: str):
       log.info(f"Building batch request file: {file_name}")
       with open(file_name, "w") as file:
           for row_id, _ in tqdm.tqdm(self.df.iterrows(), total=len(self.df)):
               for prompt_json in self._process_row_to_prompts(row_id):
                   file.write(prompt_json + "\n")
       log.info(f"Batch request file built successfully: {file_name}")
   def _process_row_to_prompts(self, idx: int) -> List[str]:
    search_query = self.df.loc[idx]["searchQuery"]
    application_title= self.df.loc[idx]["application_title"]
    products_raw = self.df.loc[idx]["products"]
    product_string_list: List[str] = []
    if isinstance(products_raw, np.ndarray):
        products_raw = products_raw.tolist()
    if isinstance(products_raw, str):
       try:
           product_string_list = json.loads(products_raw)
           if (
               not isinstance(product_string_list, list)
               or not all(isinstance(item, str) for item in product_string_list)
           ):
               log.error(f"Expected a list of JSON strings, got: {products_raw[:200]}")
               product_string_list = []
       except json.JSONDecodeError as e:
           log.error(f"Error decoding products JSON string: {e}, Content: {products_raw}",exc_info=True)
           product_string_list = []
    elif isinstance(products_raw, list):
       product_string_list = products_raw
    else:
       log.warning(f"Unexpected type for 'products' column: {type(products_raw)}")
    prompt_jsons: List[str] = []
    for i in range(0, len(product_string_list), self.products_per_prompt):
       product_chunk = product_string_list[i : i + self.products_per_prompt]
       prompt_messages = self.prompt_generator.generate_prompt(
           search_query,application_title, json.dumps(product_chunk), self.max_prompt_tokens
       )
       if prompt_messages:
           prompt_jsons.append(
               json.dumps(
                   {
                       "custom_id": f"{idx}_{i // self.products_per_prompt}",
                       "method": "POST",
                       "url": "/v1/chat/completions",
                       "body": {"model": self.config.model_id, "messages": prompt_messages},
                   }
               )
           )
       else:
           log.warning(f"Skipping prompt generation for row {idx}, chunk {i // self.products_per_prompt} due to errors.")
    return prompt_jsons
class GasBatchJobManager(AbstractBatchJobManager):

   def __init__(self, config: Config, bucket_service_output: CachedBucketService, file_name: str):
       super().__init__(config, bucket_service_output, file_name)
       self._access_token = self._get_access_token()
       self._file_id = None
       self._batch_id = None

   @staticmethod
   def _get_access_token():
       token_response = requests.post(
           f"{gas_config.apigee_base_url}/oauthv2/token",
           data={
               "grant_type": "client_credentials",
               "client_id": gas_config.client_id,
               "client_secret": gas_config.client_secret,
           },
       )
       if token_response.status_code != 200:
           log.error(f"Failed to get access token: {token_response.text}")
           raise Exception("Failed to get access token")
       access_token = token_response.json().get("access_token")
       if not access_token:
           log.error("Access token not found in token response")
           raise Exception("Access token not found")
       return access_token

   def _generate_request_data(self, file_path: str) -> Generator[str, None, None]:
       """
       Generator to read the JSONL file and yield individual request data.
       This avoids loading the entire file into memory.
       """
       with open(file_path, 'r') as f:
           for line in f:
               if line.strip():
                   yield line.strip()  

   def create_batch_request(self, df: pd.DataFrame, prompt_generator: PromptGenerator) -> str:
       log.info('Constructing batch request content...')
       GasBatchRequestBuilder(df, prompt_generator, self.config).build(self.file_name)
       self.bucket_service_output.add(self.file_name)

       upload_url = f"{gas_config.apigee_base_url}/genaisp/gas/batches"  
       headers = {"Authorization": f"Bearer {self._access_token}", "Content-Type": "application/json"}

       batch_id = None
       try:
           for request_data in self._generate_request_data(self.file_name):
               response = requests.post(upload_url, headers=headers, data=request_data)
               response.raise_for_status()  
               if not batch_id:  
                   batch_response = response.json()              
                   batch_id = batch_response.get('id')  

           log.info("Batch requests submitted successfully (streaming).")
           

       except requests.exceptions.RequestException as e:
           log.error(f"Error during batch request: {e}")
           raise  
       except json.JSONDecodeError as e:
           log.error(f"Error decoding API response: {e}")
           raise

       return batch_id

   def run_batch_prediction_job(self, input_path: str) -> GasBatchJobMonitor:
       log.info('Sending request for batch job...')      
       batch = requests.post(
           f"{gas_config.apigee_base_url}/genaisp/gas/batches",  
           headers={"Authorization": f"Bearer {self._access_token}"},
           json={"input_file_id": input_path, "endpoint": "/v1/chat/completions", "completion_window": "24h"},  
       )
       batch.raise_for_status()
       log.info('Batch details ↓↓↓')
       log.info(batch.json())
       batch_id = batch.json()['id']
       monitor_config = MonitorConfig(
           access_token=self._access_token,
           bucket_service_output=self.bucket_service_output,
           bq_client=bigquery.Client(),
           bq_table_id=Config.TABLE_ID
       )
       return GasBatchJobMonitor(batch_id, monitor_config)
       
def run_batch_job():
    bucket_service_output = CachedBucketService(Config.output_bucket_gspath)
    bucket_csv = CachedBucketService(Config.CSV_bucket)
    prompt_generator = PromptGenerator()
    df = get_products_from_bqm("temp")
    if df.empty:
        log.error("No product data found. Exiting.")
        return
    job_manager = GasBatchJobManager(Config, bucket_service_output, Config.batch_request_file_name)
    request_builder = GasBatchRequestBuilder(df, prompt_generator, Config, max_prompt_tokens=3800)
    if not Path(Config.batch_request_file_name).exists():
        log.info("Batch request file not found. Building...")
        request_builder.build(Config.batch_request_file_name)
    else:
        log.info("Reusing existing batch request file.")
    chunk_file_paths = list(Path('.').glob(f"{Config.batch_request_file_name.replace('.jsonl','')}_part_*.jsonl"))
    if not chunk_file_paths:
        log.info("No chunk files found. Splitting batch request file...")
        chunk_file_paths = split_jsonl_file(Config.batch_request_file_name, max_lines=1000)
    else:
        log.info(f"Found {len(chunk_file_paths)} pre-existing chunk files. Resuming...")
    processed_chunks = load_processed_chunks()
    def process_chunk(part_file_path, job_manager, df, total_chunks, processed_chunks):
        if str(part_file_path) in processed_chunks:
            log.info(f"Skipping already processed chunk: {part_file_path}")
            return
        try:
            log.info(f"Processing chunk: {part_file_path}")
            with open(part_file_path, 'rb') as f:
                file_data = f.read()
            batch_definition_file = requests.post(
                f"{gas_config.apigee_base_url}/genaisp/gas/files",
                headers={"Authorization": f"Bearer {job_manager._access_token}"},
                files={"file": (os.path.basename(part_file_path), file_data, "application/x-ndjson")},
                data={"purpose": "batch"},
            )
            batch_definition_file.raise_for_status()
            response_json = batch_definition_file.json()
            if "id" not in response_json:
                log.error(f"Missing 'id' in response for chunk: {part_file_path}")
                save_failed_chunk(part_file_path)
                return
            batch_file_id = response_json["id"]
            monitor = job_manager.run_batch_prediction_job(batch_file_id)
            result_path = monitor.monitor(refresh_time=100, df=df)
            if result_path:
                save_processed_chunk(part_file_path)
                processed_count = load_processed_chunks()
                processed_chunk_count = len(processed_count)
                log.info(f"Successfully processed and saved: {part_file_path}")
                log.info(f"Processed chunks count so far: {processed_chunk_count} / {total_chunks}")
            else:
                log.warning(f"Monitor failed, skipping save for: {part_file_path}")
                save_failed_chunk(part_file_path)
        except requests.exceptions.RequestException as e:
            if e.response.status_code == "401":
                log.error("401 unauthorized - please check your API key or auth token")
            else:
                log.error(f"Error in processing current chunk")
               
            save_failed_chunk(part_file_path)
    def run_parallel_processing(job_manager, chunk_file_paths, df, processed_chunks):
        total_chunks = len(chunk_file_paths)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_chunk, part_file_path, job_manager, df, total_chunks, processed_chunks)
                for part_file_path in chunk_file_paths
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    run_parallel_processing(job_manager, chunk_file_paths, df, processed_chunks)
    if len(load_processed_chunks()) == len(chunk_file_paths):
        clear_processed_chunks()
        bucket_service_output.add(Config.batch_request_file_name)
        cleanup_old_input_files(Config.batch_request_file_name)
        log.info("cleared all input files")
        log.info("All chunks processed. Clearing processed_chunks.json for next run.")
        try:
            log.info("Extracting full BigQuery table to CSV...")
            bq_client = bigquery.Client()
            table_id = Config.TABLE_ID
            today_str= date.today()
            formatted_date= today_str.strftime('%Y-%m-%d')
            destination_file = f"full_batch_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            query = f"SELECT * FROM `{table_id}` where Load_date >= '{formatted_date}'"
            df = bq_client.query(query).to_dataframe()
            df.to_csv(destination_file, index=False)
            log.info(f"Exported table to local CSV: {destination_file}")
            bucket_csv.add(destination_file)
            log.info(f"Uploaded final CSV to GCS: {destination_file}")
            log.info("All chunks processed. Clearing processed_chunks.json for next run.")
        except Exception as e:
            log.error(f"Failed to extract and upload BigQuery table to CSV: {e}", exc_info=True)
   
    else:
        log.warning("Not all chunks succeeded — retry again.")
if __name__ == "__main__":
    run_batch_job()