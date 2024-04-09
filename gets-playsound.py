import sys
import os
from gtts import gTTS
from playsound import playsound

def generate_mp3(text):
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save('output.mp3')

# Function to iterate over all .txt files in a directory and convert them to mp3 files
def convert_to_mp3():
    for filename in os.listdir(sys.argv[1]):
        if filename.endswith('.txt'):
            with open(os.path.join(sys.argv[1], filename), 'r') as file:
                text = file.read()
            generate_mp3(text)
            playsound('output.mp3')

# Call the function to convert .txt files to mp3
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script_name.py directory_path")
    else:
        convert_to_mp3()
