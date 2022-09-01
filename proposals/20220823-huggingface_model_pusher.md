#### SIG TFX-Addons
# Project Proposal

---

**Your name:** Chansung Park

**Your email:** deep.diver.csp@gmail.com

**Your company/organization:** Individual ([ML GDE](https://developers.google.com/community/experts/directory/profile/profile-chansung-park))

**Project name:** HuggingFace Model Pusher

## Project Description
HuggingFace Model Pusher(`HFModelPusher`) pushes blessed model to the [HuggingFace Model Hub](https://huggingface.co/models). Also, it optionally pushes application to [HuggingFace Space Hub](https://huggingface.co/spaces).

## Project Category
Component

## Project Use-Case(s)
The HuggingFace Model and Space Hubs let us have [Git-LFS](https://git-lfs.github.com) enabled repositories in public and private modes. Since these repositories are primarily based on Git, they're easier as far as version control is concerned. 

Supported models hosted on the HuggingFace Model Hub can be directly loaded/used with APIs provided by [transformers](https://huggingface.co/docs/transformers/index) package. However, it is not limited. We can host arbitrary types of models too. 

HuggingFace Space Hub provides free resources to host prototype applications that use machine learning models. It supports both private and public modes. Currently supported application frameworks are Gradio and Streamlit. It is often a good idea to host the current version of the model to the HuggingFace Space, so it can be tested before the production deployment. But you can host your models HuggingFace Spaces for other purposes as well.

By keeping these information in mind, `HFModelPusher` let you push a trained or blessed model to the HuggingFace Model Hub within a new branch within TFX pipeline. Then, if specified, it pushes an application to the HuggingFace Space Hub by injecting the current model information into the prepared template sources.

## Project Implementation
HFModelPusher is a class-based TFX component, and it inherits from TFX standard `Pusher` component.

### Behaviour

1. Creates HuggingFace Hub __Model__ repository. It will clone one if there is already existing repository
2. Checks out a new branch with the name of current model version. Since the model is pushed for experimental purpose, it would be good to track the versions of the model within separate branches (When the model is ready to be open to public, one can manually merge the right version(branch) into the main branch)
3. Download all the model related files(i.e. `SavedModel`) generated from the upstream component such as `Trainer` into the model repository directory. The model files could be stored in GCS bucket, so `tf.io.gfile` module is a good choice since it handles files in location agnostic manner (GCS or local)
4. Add & commit the current status of the Model repository
5. Pushes the commit to the remote HuggingFace Model repository

6. Creates HuggingFace Hub __Space__ repository. It will clone one if there is already an existing repository
7. Copy all the application related files(in `app_path`) into the space repository directory 
8. Replace special tokens in every files under the repository directory.
    - there are two special tokens of $HF_MODEL_REPO_NAME and $HF_MODEL_REPO_BRANCH. Each special tokens(placeholders) will be replaced by the id and branch of the model repository in the step 5. 
9. Add & commit the current status of the Space repository
10. Pushes the commit to the __main__ branch of the HuggingFace Space Repository
    - only the app on the main branch can be built and run, so the branch for the Space repository should not be managed


### Structure
It takes the following inputs:
```
HFModelPusher(
    username: str,
    hf_access_token: str,
    repo_name: Optional[str],
    hf_space_config: Optional[HFSpaceConfig] = None,
    model: Optional[types.Channel] = None,
    model_blessing: Optional[types.Channel] = None,    
)
```
- `username` : username of the HuggingFace user (can be an individual user or an organization)
- `hf_access_token` : access token value of the HuggingFace user. 
- `repo_name` : the repository name to push the current version of the model to. The default value is same as the TFX pipeline name
- `model` : the model artifact from the upstream TFX component such as `Trainer`
- `model_blessing` : the blessing artifact from the upstream TFX component such as `Evaluator`

It gives the follwing outputs:
- `pushed` : integer value to denote if the model is pushed or not. This is set to 0 when the input model is not blessed, and it is set to 1 when the model is successfully pushed
- `pushed_version` : string value to indicate the current model version. This is decided by `time.time()` Python built-in function
- `repo_id` : model repository ID where the model is pushed to. This follows the format of f"{username}/{repo_name}"
- `branch` : branch name where the model is pushed to. The branch name is automatically assigned to the same value of  `pushed_version`
- `commit_id` : the id from the commit history (branch name could be sufficient to retreive a certain version of the model) of the model repository
- `repo_url` : model repository URL. It is something like f"https://huggingface.co/{repo_id}/{branch}"
- `space_url` : space repository URL. It is something like f"https://huggingface.co/{repo_id}"

### HuggingFace Space spacific configurations
```
HFSpaceConfig(
    app_path: str,
    repo_name: Optional[str],
    space_sdk: Optional[str] = "gradio",
    placeholders: Optional[Dict] = None
)

# default placeholders
# the keys should be used as is. the values can be 
# changed as needed. if so, make sure there are the
# same strings in the files under `app_path`
{
    "model_repo": "$HF_MODEL_REPO_NAME",
    "model_branch": "$HF_MODEL_REPO_BRANCH"
}
```
- `app_path` : path where the application templates are in the container that runs the TFX pipeline. This is expressed either apps.gradio.img_classifier or apps/gradio.img_classifier
- `repo_name` : the repository name to push the application to. The default value is same as the TFX pipeline name
- `space_sdk` : either `gradio` or `streamlit`. this will decide which application framework to be used for the Space repository. The default value is `gradio`
- `placeholders` : dictionary which placeholders to replace with model specific information. The keys represents describtions, and the values represents the actual placeholders to replace in the files under the `app_path`. There are currently two predefined keys, and if `placeholders` is set to `None`, the default values will be used.

## Project Dependencies
- [tfx](https://pypi.org/project/tfx/)
- [huggingface-hub](https://pypi.org/project/huggingface-hub/)

## Project Team
- Chansung Park, @deep-diver, deep.diver.csp@gmail.com
- Sayak Paul, @sayakpaul, spsayakpaul@gmail.com

# Note
Please be aware of the processes and requirements which are outlined here:

* [SIG-TFX-Addons](https://github.com/tensorflow/tfx-addons)
* [Contributing Guidelines](https://github.com/tensorflow/tfx-addons/blob/main/CONTRIBUTING.md)
* [TensorFlow Code of Conduct](https://github.com/tensorflow/tfx-addons/blob/main/CODE_OF_CONDUCT.md)