#### SIG TFX-Addons
# Project Proposal

---

**Your name:** Chansung Park

**Your email:** deep.diver.csp@gmail.com

**Your company/organization:** Individual ([ML GDE](https://developers.google.com/community/experts/directory/profile/profile-chansung-park))

**Project name:** HuggingFace Space Pusher

## Project Description
HuggingFace Space Pusher(`HFSpacePusher`) pushes a model to the [HuggingFace Space Hub](https://huggingface.co/spaces).

## Project Category
Component

## Project Use-Case(s)
The HuggingFace Space Hub lets us have [Git-LFS](https://git-lfs.github.com) enabled repositories in public and private modes. It comes with free resources to host a prototype application that uses machine learning models. Currently supported application frameworks are [Gradio](https://gradio.app) and [Streamlit](https://streamlit.io). 

It is often a good idea to host a currently built model to the Huggingface Space, so it could be interated with real world before the production deployment. Also, HuggingFace Space Hub is easy to manage file versions, especially for those familiar with Git. It is not common, but models files along with application also can be managed within the same Space repository.

## Project Implementation
HFSpacePusher is a class-based TFX component, and it takes the following inputs:

```
HFModelPusher(
    username: str,
    hf_access_token: str,
    app_path: str,
    repo_name: Optional[str],
    space_sdk: Optional[str],
    model_hub_repo_id: Optional[str],
    model_hub_repo_branch: Optional[str],
    options: [Optional[Dict[str, str]]],
    model: Optional[types.Channel],
    model_blessing: Optional[types.Channel],
)
```
- `username` : username of the HuggingFace user (can be an individual user or an organization)
- `hf_access_token` : access token value of the HuggingFace user.
- `app_path`: path where the application templates are in the container that runs the TFX pipeline. This is expressed either `apps.gradio.img_classifier` or `apps/gradio.img_classifier`
- `repo_name` : the repository name to push the application to. The default value is same as the TFX pipeline name
- `space_sdk`: either `gradio` or `streamlit`. This will decide which application framework to be used for the Space repository
- `model_hub_repo_id`: name of HuggingFace Model repository to be injected into the files in `app_path`
- `model_hub_repo_branch`: branch of HuggingFace Model repository to be injected into the files in `app_path`
- `model` : the model artifact from the upstream TFX component such as `Trainer`
- `model_blessing` : the blessing artifact from the upstream TFX component such as `Evaluator`
- `options` : optional key/value pairs. key denotes placeholders in string, and values denotes string to replace the placeholders

It gives the follwing outputs:
- `pushed` : integer value to denote if the space is pushed or not. 
- `pushed_version` : string value to indicate the current space version. This is decided by `time.time()` Python built-in function
- `repo_id` : repository ID where the space is pushed to. This follows the format of `f"{username}/{repo_name}"`
- `commit_id` : the id from the commit history
- `repo_url` : repository URL. It is something like f"https://huggingface.co/{repo_id}/"

The behaviour of the component:
1. Creates HuggingFace Hub Repository object using the `huggingface-hub` package. It will clone one if there is already an existing repository
2. Copy all the application related files(in `app_path`) into the directory of the Repository from the step 1.
3. if `model_hub_repo_id` isn't set but `model`, the model from the upstream component such as `Trainer` will be downloaded into the repository directory.
    - in this case, a function will be auto-generated with the special name like `get_model()` which loads and returns the model in the same repository
4. if `model` isn't set but `model_hub_repo_id`, it will replace special tokens in every files under the repository directory.
    - in this case, the component should be used after the `HFModelPusher` which is another custom TFX component.
    - there are two special tokens of `$MODEL_REPO_ID` and `$MODEL_VERSION`. Each special tokens(placeholders) will be replaced by `model_hub_repo_id` and `model_hub_repo_branch`.
5. Add & commit the current status (files)
6. Pushes the commit to the main branch of the HuggingFace Model Repository
    - only the app on the main branch can be built and run, so the branch for the Space repository should not be managed

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