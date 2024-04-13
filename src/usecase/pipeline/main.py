from deploy import Deploy


def main():
    print("Deploying the model")

if __name__ == "__main__":
    deploy = Deploy()
    deploy.create_s3()
    deploy.get_llm_image_uri()
    deploy.deploy_model()