# Guidelines for Contributing

PyPolo values the contributions of interested individuals or groups.
The following guidelines provide information on how you can contribute to the library:

There are four primary ways to contribute to PyPolo, listed below in descending order of difficulty:

- 🔧 Adding or improved **functionality** to the existing codebase.
- 🐛 **Fixing issues or bugs** with the existing codebase.
- 📖 Adding or improving the **documentation and examples**.
- 🙏 **Submitting issues** related to bugs or desired enhancements.

We look forward to reviewing your contributions and thank you for your interest in improving PyPolo!

## Pull Request Guidelines

The first three types of contributions (adding or fixing code and documentation) can be submitted by creating a pull request.
Here's an overview of the process for creating a pull request:

- **[Fork](https://github.com/Weizhe-Chen/PyPolo/fork)** the repository and create a branch

    Create a branch with a descriptive name for your changes. This will help us understand the purpose of your pull request.

- **Make the changes**

    Make your changes to the codebase. Please ensure that your code is well-written and follows our code style guidelines.

- Submit the **[Pull Request](https://github.com/Weizhe-Chen/PyPolo/pulls)**

    Submit your pull request with a clear and concise title and description. Please include any relevant information or context that may be helpful in reviewing your changes.

- **Respond** to feedback

    We may ask for changes to be made to your pull request. Please respond to feedback promptly and make any necessary changes.

- **Merge** the pull request

    Once your pull request has been reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!

## Issues Submitting Guidelines

- **Search for [existing issues](https://github.com/Weizhe-Chen/PyPolo/issues)**

    Before submitting a new issue, search the repository to see if the issue has already been reported.
    If you find an existing issue that describes the same problem you are having, add a comment with any additional information you can provide.

- Provide a **clear title**

    Use a clear and concise title that summarizes the problem you are experiencing.
    Avoid using generic titles like "Bug" or "Problem."

- **Describe** the issue

    Provide a detailed description of the issue you are experiencing, including any error messages or other relevant information.
    Include steps to reproduce the issue, if possible.

- Provide **relevant context**

    Provide any relevant context that could help developers understand the issue, such as your operating system, browser version, or other software you are using.

- Provide a **proposed solution**

    If you have an idea for how to solve the issue, include it in your description.
    Even if you're not a developer, you might have ideas or suggestions that can help.

- Use **Markdown**

    Use Markdown to format your issue description, including headings, bullet points, and code blocks.
    This will make your issue easier to read and understand.

- Be polite and respectful

    Remember that developers are people too, and they are volunteering their time to work on the project.
    Be polite and respectful in your interactions with them.

By following these guidelines, you can help ensure that your issue is clear, actionable, and helps developers resolve the issue as quickly as possible.

## Python Project Code Style Guidelines

These guidelines aim to help maintain a consistent and high-quality codebase.
Please follow these guidelines when submitting code changes to the repository.

### Code Formatting

- Use double quotes for string literals.
- Use [yapf](https://github.com/google/yapf) to format the code.
- Use [isort](https://pycqa.github.io/isort/) to sort the imports

### Naming Conventions

- Use descriptive names for variables, functions, classes, and exceptions.
- Avoid using abbreviations or acronyms in names.
- Prefix private variables and functions with an underscore (`_`).
- Use `snake_case` for variables and functions.
- Use `CamelCase` for classes and exceptions.
- Use `UPPER_CASE` for constants.

### Code Organization

- Use a modular approach to organize code into logical units.
- Group related functions and variables into classes or modules.
- Keep files small and focused on a single responsibility.

### Code Quality

- Use comments to provide context and explain complex code.
- Docstrings follow [Google style](https://google.github.io/styleguide/pyguide.html)
- Write code that is easy to read and understand.
- Write code that is easy to modify and maintain.
- Avoid using global variables and functions.
- Write [pytest](https://docs.pytest.org/en/7.2.x/) unit tests for all code changes.
- Use type hints to improve readability and maintainability.
- Use [pyright](https://github.com/microsoft/pyright) for static type checking.

### Exception Handling

- Use exception handling to handle errors and unexpected situations.
- Define custom exceptions for specific error conditions.
- Raise exceptions to signal error conditions.
- Use context managers for resource management.

### Git Practices

- Use descriptive commit messages.
- Use feature branches and pull requests for code changes.
- Keep commits small and focused on a single change.
- Rebase and merge instead of merging with merge commits.


### Development Tools

Below is a summary of the development tools and their purposes.

| Name | Functionality |
| ---- | ------------- |
| [yapf](https://github.com/google/yapf) | Automatic code formatting |
| [isort](https://pycqa.github.io/isort/) | Automatic import sorting |
| [pytest](https://docs.pytest.org/en/7.2.x/) | Unit testing |
| [pyright](https://github.com/microsoft/pyright) | Linting |

These tools can be installed by

```bash
pip install -r requirements_dev.txt
```

Following these guidelines will help us ensure that the PyPolo codebase remains consistent, maintainable, and of high quality.
