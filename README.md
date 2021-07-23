# TFX Addons

[![TFX Addons package CI](https://github.com/tensorflow/tfx-addons/workflows/TFX%20Addons%20package%20CI/badge.svg?branch=main)](https://github.com/tensorflow/tfx-addons/actions?query=branch%3Amain)

## TL;DR
This is the repo for projects organized under the special interest group, SIG TFX-Addons. Join the group by [joining the Google Group](http://goo.gle/tfx-addons-group) and participate in projects to build new components, libraries, tools, and other useful additions to TFX.

## Context
Machine learning in production environments is a mission-critical part of a growing number of products and services across many industries. To become an [AI First company](https://ai.google/), Google required a state-of-the-art production ML infrastructure framework, and created TensorFlow Extended (TFX). Google open-sourced TFX in 2019 to enable other developers worldwide to benefit from and help us improve the TFX framework, and established open layers within the TFX architecture specifically focused at customization for a wide range of developer needs. These include custom pipeline components, containers, templates, and orchestrator support.

In order to accelerate the sharing of community customizations and additions to TFX, the TFX team would like to encourage, enable, and organize community contributions to help continue to meet the needs of production ML, expand the vision, and help drive new directions for TFX and the ML community. 

## Goals & Objectives 
We welcome community contributions on any area of TFX, but this SIG will initially focus on the following goals:

- Driving the development of high-quality custom pipeline components, including Python function-based components, container-based components, and fully custom components.
- Shaping a standardized set of descriptive metadata for community-contributed components to enable easy understanding, comparison, and sharing of components during discovery.
- Similarly driving the development of templates, libraries, visualizations, and other useful additions to TFX.

These projects will begin as proposals to the SIG, and upon approval will be led and maintained by the community members involved in the project and assigned a project folder, with high-level consultation from the TFX team.

### In-Scope, Out of Scope
Although TFX is an open-source project and we welcome contributions to TFX itself, **this SIG does not include contributions or additions to core TFX**.  It is focused only on building community-contributed and maintained additions on top of core TFX.  [Core TFX has its own repo](https://github.com/tensorflow/tfx), and PRs and issues will continue to be managed there. **In addition, all contributions must not violate the [Google AI Principles](https://ai.google/principles/) or [Responsible AI Practices](https://ai.google/responsibilities/responsible-ai-practices/).**

## Membership
We encourage any developers working in production ML environments, infrastructure, or applications to [join and participate in the activities of the SIG](http://goo.gle/tfx-addons-group). Whether you are working on advancing the platform, prototyping or building specific applications, or authoring new components, templates, libraries, and/or orchestrator support, we welcome your feedback on and contributions to TFX and its tooling, and are eager to hear about any downstream results, implementations, and extensions. 

We have multiple channels for participation, and publicly archive discussions in our user group mailing list: 
- tfx-addons@tensorflow.org – Google group for SIG TFX-Addons
- tfx@tensorflow.org – our general mailing list for TFX

Other Resources 
- SIG Repository: http://github.com/tensorflow/tfx-addons (this repo)
- Documentation: https://www.tensorflow.org/tfx

## Organization and Governance
This is the repo for individual SIG projects and contributions.  It also contains overall SIG documents and resources, which are managed by the TensorFlow team.  Individual contribution projects will begin as proposals to the SIG, and once approved a folder will be created for the project, and project leaders assigned permissions to manage the folder.  **Projects will be led, maintained, and be the responsibility of community project leaders. Google and the TensorFlow team will not provide user support or maintenance for contributed addons. The TFX team will support community maintainers in SIG operations and contribution infrastructure.**

Individual projects will be assigned a new folder, where all project materials will live. For all community-contributed projects the source of truth will be those project folders. Project leaders will be identified in CODEOWNERS. New project leaders will be recruited for abandoned projects, or if new leaders are not found then projects will be deprecated and archived. Statistics will be generated and reported per-project.

SIG TFX-Addons is a community-led open source project. As such, the project depends on public contributions, bug fixes, and documentation. This project adheres to the [TensorFlow Code of Conduct](https://github.com/tensorflow/tensorflow/blob/master/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Project Approvals
1. Project proposals will be submitted to the SIG and published for open review and comment by SIG members for 2 weeks.
2. Following review and approval by the Google TFX team, core team members will vote either in person or offline on whether to approve or reject project proposals.
3. All projects must meet the following criteria:
   - Team members must be named in the proposal
   - All team members must have completed a [Contributor License Agreement](https://cla.developers.google.com/)
   - The project must not violate the [TensorFlow Code of Conduct](https://github.com/tensorflow/tensorflow/blob/master/CODE_OF_CONDUCT.md), [Google AI Principles](https://ai.google/principles/) or [Responsible AI Practices](https://ai.google/responsibilities/responsible-ai-practices/).
4. Projects must code to supported open interfaces only, and not reach into core TFX to make changes or rely on private classes, methods, properties, or interfaces.
5. **Google retains the right to reject any proposal.**
6. Projects must first be approved by the Google team.  Projects are then sent for approval to the core community team.  Projects will be approved with a minimum of three `+1` votes, but can be sent for changes and re-review with a single `-1` vote.

## Contacts
- Project Lead:
  - Robert Crowe (Google)
- Community Lead(s)
  - Hannes Hapke (Digits)
- Core Team Members:
  - Paul Selden (OpenX)
  - Gerard Casas Saez (Twitter)
  - Newton Le (Twitter)
  - Marc Romeyn (Spotify)
  - Ryan Clough (Spotify)
  - Samuel Ngahane (Spotify)
  - Michal Brys (OpenX)
  -  Baris Can Durak (ZenML)
  -  Hamza Tahir (ZenML)
  -  Larry Price (OpenX)
- Administrative questions: 
  - Thea Lamkin (Google): thealamkin at google dot com 
  - Joana Carrasqueira (Google): joanafilipa at google dot com
  - tf-community at tensorflow dot org

Meeting cadence:
- TBD
