import sagemaker
import boto3
import json

from sagemaker.huggingface import HuggingFaceModel

class Deploy:

    def __init__(self):
        self.sagemaker = sagemaker.Session()
        self.session = boto3.Session(profile_name='default', region_name='us-east-1')

        self.role = None
        self.llm_image = None

    def deploy_model(self):
        # sagemaker config
        instance_type = "ml.g4dn.xlarge"
        number_of_gpu = 8
        health_check_timeout = 300

        # Define Model and Endpoint configuration parameter
        config = {
            'HF_MODEL_ID': "mistralai/Mixtral-8x7B-Instruct-v0.1", # model_id from hf.co/models
            'SM_NUM_GPUS': json.dumps(number_of_gpu), # Number of GPU used per replica
            'MAX_INPUT_LENGTH': json.dumps(24000),  # Max length of input text
            'MAX_BATCH_PREFILL_TOKENS': json.dumps(32000),  # Number of tokens for the prefill operation.
            'MAX_TOTAL_TOKENS': json.dumps(32000),  # Max length of the generation (including input text)
            'MAX_BATCH_TOTAL_TOKENS': json.dumps(512000),  # Limits the number of tokens that can be processed in parallel during the generation
            # ,'HF_MODEL_QUANTIZE': "awq", # comment in to quantize not supported yet
        }

        # create HuggingFaceModel with the image uri
        llm_model = HuggingFaceModel(
            role=self.role,
            image_uri=self.llm_image,
            env=config
        )

        # Deploy model to an endpoint
        # https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#sagemaker.model.Model.deploy
        llm = llm_model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            container_startup_health_check_timeout=health_check_timeout, # 10 minutes to be able to load the model
        )

    def create_s3(self):
        # sagemaker session bucket -> used for uploading data, models and logs
        # sagemaker will automatically create this bucket if it not exists
        sagemaker_session_bucket=None
        if sagemaker_session_bucket is None and self.sagemaker is not None:
            # set to default bucket if a bucket name is not given
            sagemaker_session_bucket = self.sagemaker.default_bucket()

        try:
            self.role = sagemaker.get_execution_role()
        except ValueError:
            iam = boto3.client('iam')
            self.role = iam.get_role(RoleName='sagemaker-role')['Role']['Arn']

        self.sagemaker = sagemaker.Session(default_bucket=sagemaker_session_bucket)

        print(f"sagemaker role arn: {self.role}")
        print(f"sagemaker session region: {self.sagemaker.boto_region_name}")

    def get_llm_image_uri(self):
        region_mapping = {
            "af-south-1": "626614931356",
            "il-central-1": "780543022126",
            "ap-east-1": "871362719292",
            "ap-northeast-1": "763104351884",
            "ap-northeast-2": "763104351884",
            "ap-northeast-3": "364406365360",
            "ap-south-1": "763104351884",
            "ap-south-2": "772153158452",
            "ap-southeast-1": "763104351884",
            "ap-southeast-2": "763104351884",
            "ap-southeast-3": "907027046896",
            "ap-southeast-4": "457447274322",
            "ca-central-1": "763104351884",
            "cn-north-1": "727897471807",
            "cn-northwest-1": "727897471807",
            "eu-central-1": "763104351884",
            "eu-central-2": "380420809688",
            "eu-north-1": "763104351884",
            "eu-west-1": "763104351884",
            "eu-west-2": "763104351884",
            "eu-west-3": "763104351884",
            "eu-south-1": "692866216735",
            "eu-south-2": "503227376785",
            "me-south-1": "217643126080",
            "me-central-1": "914824155844",
            "sa-east-1": "763104351884",
            "us-east-1": "763104351884",
            "us-east-2": "763104351884",
            "us-gov-east-1": "446045086412",
            "us-gov-west-1": "442386744353",
            "us-iso-east-1": "886529160074",
            "us-isob-east-1": "094389454867",
            "us-west-1": "763104351884",
            "us-west-2": "763104351884",
        }

        self.llm_image = f"{region_mapping[self.sagemaker.boto_region_name]}.dkr.ecr.{self.sagemaker.boto_region_name}.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.3.1-gpu-py310-cu121-ubuntu20.04-v1.0"

        # print ecr image uri
        print(f"llm image uri: {self.llm_image}")
