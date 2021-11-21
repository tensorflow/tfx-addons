# Contribution Guidelines

## Directory Structure
The repo contains three main directories as follows:
- **[Component](./component):** Contains the main component code with a separate file for the executor code
- **[Data](./data):** Containing the sample data to be used for testing
- **[Example](./example):** Contains example codes to test our component with the CSVs present in [data](./data)

## A few Git and GitHub practices

### Commits
Commits serve as checkpoints during your workflow and can be used to **revert back** in case something gets messed up.
- **When to commit:** Try not to pile up many changes in multiple commits while ensuring that you don't make too many commits for fixing a small issue.
- **Commit messages:** Commit messages should be descriptive enough for an external person to get an idea of what it accomplished while ensuring they don't exceed 50 characters.

Check out [this](https://gist.github.com/turbo/efb8d57c145e00dc38907f9526b60f17) for more information about the good practices

### Branches
Branches are a good way to simulataniously work on different features at the same time. Check out [git-scm](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging) to know more about various concepts involved in the same.

For descriptive branch names, it is a good idea to follow the following format:
**`name/keyword/short-description`**
- **Name:** Name of the person/s working on the branch. This can be ignored if many people(>2) are expected to work on it.
- **Keyword:** This describes what "type" of work this branch is supposed to do. These are typically named as:
    - `feature`: Adding/expanding a feature
    - `base`: Adding boilerplate/readme/templates etc.
    - `bug`: Fixes a bug
    - `junk`: Throwaway branch created to experiment
- **Short description:** As the name suggests, this contains a short description about the branch, usually no longer than 2-3 words separated by a hyphen (`-`).

P.S. If multiple branches are being used to work on the same issue (say issue `#n`), they can be named as `name/keyword/#n-short-description`

### Issues 
The following points should be considered while creating new issues
- Use relevant labels like `bug`, `feature` etc.
- If the team has decided the person who will work on it, it should be **assigned** to the said person as soon as possible to prevent same work being done twice.
- The issue should be linked in the **project** if needed and the status of the same should be maintained as the work progresses.

### Pull Requests
It is always a good idea to ensure the following are present in your Pull Request description:
- Relevant issue/s
- What it accomplished
- Mention `[WIP]` in title and make it a `Draft Pull Request` if it is a work in progress
- Once the pull request is final, it should be **requested for review** from the concerned people
