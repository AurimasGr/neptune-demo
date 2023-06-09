# AzureML how-to

This project is an example integration between Azure DevOps and Azure ML services with Neptune.

## Environment preparation

### AzureML

Before you can run the example, you will first need to create a compute cluster and a custom environment in AzureML environment. You can do that by executing scripts 
```./dependencies/build_compute_cluster.py``` and ```./dependencies/build_environment.py```.

Note that you will need to fill 

```
AZURE_SUBSCRIPTION_ID = "<YOUR SUBSCRIPTION ID>"
AZUREML_RESOURCE_GROUP_NAME = "<YOUR RESOURCE GROUP NAME>"
AZUREML_WORKSPACE_NAME = "<YOUR WORKSPACE NAME>"
```
with values representing your environment.

### Azure DevOps

For Azure DevOps Pipelines to be able to successfully create and execute AzureML Pipelines, you will need to create the following secrets as per ```./azure-ci/azure-pipelines.yaml``` in your Azure DevOps Pipeline via UI:

```
AZURE_TENANT_ID: $(tenant)
AZURE_CLIENT_ID: $(client)
AZURE_CLIENT_SECRET: $(secret)
NEPTUNE_API_TOKEN: $(neptune-sa-token)
```

## The example
The example is focused around creation of Azure DevOps CI/CD pipeline that would be able to test the AzureML Pipeline and then deploy it for operational purposes. The following picture represents resulting AzureML pipeline:

![Screenshot 2023-04-17 at 16.32.20.png](..%2F..%2FScreenshot%202023-04-17%20at%2016.32.20.png)