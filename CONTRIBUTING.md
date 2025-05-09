# Contributing to SMT

*Thanks for checking out this project!*

This project is part of an academic and engineering effort to build, evaluate, and extend deep 
neural network models. We appreciate all contributions, whether they involve fixing bugs, 
improving documentation, enhancing reproducibility, or proposing new experimental ideas.

The following guide outlines how to get started, what we expect from contributions, and 
how to make your pull request easy to review and merge. Please read it carefully before 
submitting your changes.

## :pushpin: Before You Start

- Check the [Issues](https://github.com/antoniorv6/SMT/issues) tab to avoid duplicate work.
- If you're proposing a major change, open an issue first to discuss your idea.
- Before [creating an issue](https://help.github.com/en/github/managing-your-work-on-github/creating-an-issue), check if you are using the latest version of the project.
- Prefer using [reactions](https://github.blog/2016-03-10-add-reactions-to-pull-requests-issues-and-comments/), not comments, if you simply want to "+1" an existing issue.
- Please follow the coding, documentation, and testing conventions described below.

## :hammer_and_wrench: How to Contribute

#### 1. Fork the Repository
#### 2. Clone Your Fork

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

#### 3. Set Up the Environment

Follow recommendations on the main [README](https://github.com/antoniorv6/SMT/#project-setup), set-up your envirnment either using UV or Docker.

#### 4. Create a Branch

```bash
git switch -c fix/issue-description
```

#### 5. Write code (& commits)

- Model code lives in `smt_model`
- Configurations in `config`
- Tests in `test`
- Launch scripts and utilities in the root folder

#### 6. Push an Open a Pull Request

```bash
git push origin fix/issue-description
```

From the GitHub interface, [submit your pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork) to the repo.

## :envelope: Submitting Pull Requests

*Note: All contributions will be licensed under the project's license.*

- **Smaller is better.** Submit **one** pull request per bug fix or feature. A pull request should contain isolated changes pertaining to a single bug fix or feature implementation. **Do not** refactor or reformat code that is unrelated to your change. It is better to **submit many small pull requests** rather than a single large one. Enormous pull requests will take enormous amounts of time to review, or may be rejected altogether.

- **Test before opening requests.** Please, verify that the results you get with your branch are similar to the ones presented in the paper. If your solution improves the paper, please **notify the support team to verify and include your results**. This is an open-source project, so all performance improvements will be credited to the author of the commit.

- **Coordinate bigger changes.** For large and non-trivial changes, open an issue to discuss a strategy with the maintainers. Otherwise, you risk doing a lot of work for nothing!

- **Prioritize understanding over cleverness.** Write code clearly and concisely. Remember that source code usually gets written once and read often. Ensure the code is clear to the reader. The purpose and logic should be obvious to a reasonably skilled researcher, otherwise you should add a comment that explains it.

- **Add documentation.** Document your changes with code doc comments or in existing guides.

- **Update the CHANGELOG** for all enhancements and bug fixes. Include the corresponding issue number if one exists, and your GitHub username. (example: "- Fixed crash in profile view. #123 @jessesquires")

- **Use the repo's default branch.** Branch from and [submit your pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork) to the repo's `master` branch.

- **[Resolve any merge conflicts](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/resolving-a-merge-conflict-on-github)** that occur.

## :memo: Writing Pull Requests

1. Limit the subject to a short clear line
1. Use the imperative mood in the subject line (example: "Fix input preprocessing")
1. Keep the body simple, avoid large unecesary bodies
1. Use the body to explain **why**, *not what and how* (the code shows that!)
1. Label pull requests with the appropiated tag (bug, enhancement...)

```
TITLE: Short summary of changes

BODY:
Add a more detailed explanation here, if necessary. Wrap it to about 72
characters or so. Possibly give some background about the issue being
fixed, etc. The body of the commit message can be several paragraphs.
Further paragraphs come after blank lines and please do proper word-wrap.

Explain the problem that this commit is solving. Focus on why you
are making this change as opposed to how or what. The code explains 
how or what. Reviewers and your future self can read the patch, 
but might not understand why a particular solution was implemented.
Are there side effects or other unintuitive consequences of this
change? Here's the place to explain them.

 - Bullet points are okay, too

 - A hyphen or asterisk should be used for the bullet, preceded
   by a single space, with blank lines in between

Note the fixed or relevant GitHub issues at the end:

Resolves: #123
See also: #456, #789
```

## :memo: Writing Commits

1. The commit description must reflect the
1. If a issue is being solved in the commit, add it to the bodie with its ID
1. A clean style, like [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/#summary), allows to review your code better and faster

## :medal_sports: Certificate of Origin

*'s Certificate of Origin 1.1*

By making a contribution to this project, I certify that:

> 1. The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
> 1. The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or
> 1. The contribution was provided directly to me by some other person who certified (1), (2) or (3) and I have not modified it.
> 1. I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
