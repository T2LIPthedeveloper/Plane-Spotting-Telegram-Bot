import logging

from telegram import __version__ as TG_VER
from multiprocessing import Process, Queue

idx_to_names = {
  0 : "Boeing 707-320", 
  1 : "Boeing 727-200",
  2 : "Boeing 737-200",
  3 : "Boeing 737-300",
  4 : "Boeing 737-400",
  5 : "Boeing 737-500",
  6 : "Boeing 737-600",
  7 : "Boeing 737-700",
  8 : "Boeing 737-8",
  9 : "Boeing 737-800",
  10 : "Boeing 737-9",
  11 : "Boeing 737-900",
  12 : "Boeing 747-100",
  13 : "Boeing 747-200",
  14 : "Boeing 747-300",
  15 : "Boeing 747-400",
  16 : "Boeing 747-8",
  17 : "Boeing 757-200",
  18 : "Boeing 757-300",
  19 : "Boeing 767-200",
  20 : "Boeing 767-300",
  21 : "Boeing 767-400",
  22 : "Boeing 777-200",
  23 : "Boeing 777-300",
  24 : "Boeing 787-10",
  25 : "Boeing 787-8",
  26 : "Boeing 787-9",
  27 : "Airbus A300B4",
  28 : "Airbus A310",
  29 : "Airbus A318",
  30 : "Airbus A319",
  31 : "Airbus A320",
  32 : "Airbus A321",
  33 : "Airbus A330-200",
  34 : "Airbus A330-300",
  35 : "Airbus A340-200",
  36 : "Airbus A340-300",
  37 : "Airbus A340-500",
  38 : "Airbus A340-600",
  39 : "Airbus A350-1000",
  40 : "Airbus A350-900",
  41 : "Airbus A380",
  42 : "ATR-42",
  43 : "ATR-72",
  44 : "British Aerospace 146-200",
  45 : "British Aerospace 146-300",
  46 : "British Aerospace 125",
  47 : "Boeing 717-200",
  48 : "Lockheed Martin C-130",
  49 : "Bombardier CRJ-200",
  50 : "Bombardier CRJ-700",
  51 : "Bombardier CRJ-900",
  52 : "Cessna 172",
  53 : "Cessna 208",
  54 : "Cessna 525",
  55 : "Cessna 560",
  56 : "Bombardier Challenger 600",
  57 : "McDonnell Douglas DC-10",
  58 : "McDonnell Douglas DC-8",
  59 : "McDonnell Douglas DC-9-30",
  60 : "Embraer E-170",
  61 : "Embraer E-190",
  62 : "Embraer E-195",
  63 : "Embraer ERJ 135",
  64 : "Embraer ERJ 145",
  65 : "Embraer Legacy 600",
  66 : "Dassault Falcon 2000",
  67 : "Dassault Falcon 900",
  68 : "Fokker 100",
  69 : "Fokker 50",
  70 : "Fokker 70",
  71 : "Bombardier Global Express",
  72 : "Gulfstream IV",
  73 : "Gulfstream V",
  74 : "Ilyushin Il-76",
  75 : "Lockheed L-1011 Tristar",
  76 : "McDonnell Douglas MD-11",
  77 : "McDonnell Douglas MD-80",
  78 : "McDonnell Douglas MD-87",
  79 : "McDonnell Douglas MD-90"
}

try:
  from telegram import __version_info__
except ImportError:
  __version_info__ = (0, 0, 0, 0, 0)  # type: ignore[assignment]

if __version_info__ < (20, 0, 0, "alpha", 1):
  raise RuntimeError(
      f"This example is not compatible with your current PTB version {TG_VER}. To view the "
      f"{TG_VER} version of this example, "
      f"visit https://docs.python-telegram-bot.org/en/v{TG_VER}/examples.html")

from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
import tensorflow as tf
import numpy as np
from dotenv import load_dotenv
import cv2
import os

load_dotenv()
my_bot_token = os.getenv('BOT_TOKEN')
queue = Queue()

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  """Send a message when the command /start is issued."""
  user = update.effective_user
  await update.message.reply_html(
      rf"Hi {user.mention_html()}!",
      reply_markup=ForceReply(selective=True),
  )

async def help_command(update: Update,
                       context: ContextTypes.DEFAULT_TYPE) -> None:
  """Send a message when the command /help is issued."""
  await update.message.reply_text("Help!")

async def process_image(image_path):
  try:
    img = cv2.imread(image_path)
    img = cv2.resize(img, (299, 299))
    img = img / 255.0
    # convert to float32
    img = img.astype(np.float32)
    return img
  except Exception as e:
    logger.error("Error processing image: %s", e)
    return None

def run_tflite_model(input_data, q):
    try:
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], [input_data])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        actual_output = output_data[0]
        q.put(actual_output)
        logger.info("Output data: %s", max(actual_output))
    except Exception as e:
        logger.error("Error running TFLite model: %s", e)
        q.put(None)

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  # if the user types a non-image message, reply that you need to upload an image to predict
  replies = [
    "Please upload an image of a plane to predict the model and type.",
    "I need an image to predict the model and type of the plane.",
    "I can only predict the model and type of a plane from an image. Please upload one.",
    "That wasn't an image. Please upload an image of a plane to predict the model and type.",
    "Uh oh! Not an image. Please upload an image of a plane to predict the model and type.",
    "Oops! I need an image to predict the model and type of the plane."
  ]

  await update.message.reply_text(replies[np.random.randint(0, len(replies))])

def process_output(output_data):
  # Return the plane model and type if model is above 60% confident
  if max(output_data) < 0.7:
    return "I'm not confident enough to predict the model of the plane. These are the top possibilities:\n" + "\n".join([f"{idx_to_names[i]}: {output_data[i]}" for i in np.argsort(output_data)[-5:][::-1]if output_data[i] > 0.1])
  else:
    idx = np.argmax(output_data)
    return "It's most likely a " + idx_to_names[idx] + "."

# Replace def stylize with def process, meant to process image and return plane model and type
async def process(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  # get the image and save it
  fid = update.message.photo[-1].file_id
  newFile = await context.bot.get_file(fid)
  await newFile.download_to_drive()
  # get the path of the image
  image_path = newFile.file_path
  # parse image path to get file name
  image_path = os.path.join(os.getcwd(), image_path.split("/")[-1])
  
  if not image_path or not os.path.exists(image_path):
    await update.message.reply_text("Invalid image. Please upload an image of a plane to predict the model and type.")
  
  # check if image is in jpg or jpeg format as required
  if not image_path.endswith(".jpg") and not image_path.endswith(".jpeg"):
    await update.message.reply_text("Please upload a .jpg or .jpeg image of a plane to predict the model and type.")
  # log obtained image path
  logger.info("Obtained image path: %s", image_path)
  
  # process the image
  img = await process_image(image_path)
  if img is None:
    await update.message.reply_text("Error processing image. Please try again.")
    return
  
  # run the tflite model
  result_queue = Queue()

  # Start the model process
  model_process = Process(target=run_tflite_model, args=(img, result_queue))
  model_process.start()

  # Wait for the model process to finish and get the result
  output_data = result_queue.get()
  model_process.join()

  if output_data is None:
      await update.message.reply_text("Error occurred while running the model.")
      return
  
  # process the output
  res = process_output(output_data)
  # delete the image
  os.remove(image_path)

  # remove queue
  del result_queue

  # send the result
  await update.message.reply_text(res)

# Error handler
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  """Log the error and send a message to the user."""
  logger.exception(f"Exception while handling an update: {update}")
  logger.error(context.error)

  await update.message.reply_text("An error occurred while processing the image. Please try again.")

if __name__ == "__main__":
  """Start the bot."""
  # Create the Application and pass it your bot's token.
  application = Application.builder().token(my_bot_token).build()

  # on different commands - answer in Telegram
  application.add_handler(CommandHandler("start", start))
  application.add_handler(CommandHandler("help", help_command))

  # on non command i.e message - echo the message on Telegram
  application.add_handler(
      MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
  
  # on photo upload - process the image
  application.add_handler(
      MessageHandler(filters.PHOTO, process))
  
  # on any other type of media - ask for an image
  application.add_handler(
      MessageHandler(filters.VIDEO | filters.AUDIO | filters.VOICE, echo))
  
  # log all errors
  application.add_error_handler(error_handler)

  # Run the bot until the user presses Ctrl-C
  application.run_polling(allowed_updates=Update.ALL_TYPES)