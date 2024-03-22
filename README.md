# Plane Spotting Telegram Bot
 Telegram bot for plane spotting with custom CV models built using Python.
 Built by Atul Parida using Telegram API, the python-telegram-bot library, and custom computer vision models built on Tensorflow and Keras.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This is a Telegram bot for plane spotting. It uses custom computer vision models to identify planes from images. The bot is built using Python and uses the Telegram API to communicate with users. I built this over a two-week span, from 6th March 2024 to 20th March 2024.

## Features
- **Plane Spotting**: The bot can identify planes from images using custom computer vision models. Currently, it's about 85% accurate, but I'm working on improving it with a larger dataset and more training.
- **Image Recognition**: The bot can also recognize other objects in images, but it's not as accurate as the plane spotting feature.
- **Image Editing**: The bot can perform basic image editing tasks like cropping, rotating, and resizing images. It's still a work in progress, but I'm planning to add more features in the future.

## Installation
To install the bot, you'll need to have Python 3.9 or higher installed on your system. You can install the required dependencies using pip.

First, upgrade your pip to the latest version:
```bash
pip install --upgrade pip
```

Then, install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage
To use the bot, you'll need to create a new bot on Telegram and get the API token. You can do this by talking to the BotFather on Telegram. Once you have the API token, you can add it to the `.env` file with the label `BOT_TOKEN`.

After that, you can start the bot using the following command:

```bash
python bot.py
```

The bot will start running and you can interact with it on Telegram.

## Contributing
Contributions are welcome! If you have any ideas for new features or improvements, feel free to open an issue or submit a pull request. At the moment I'm working on using a webhook rather than polling, so any help with that would be appreciated.

## License
This project is open source and available under the [MIT License](LICENSE.md).

[//]: # (README.md ends here)
```