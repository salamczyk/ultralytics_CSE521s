import argparse
import os
import json
import cv2
import openai
openai.api_key = "****"
import time
from gtts import gTTS
from ultralytics_CSE521s_main.ultralytics import YOLO
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

def message_generate(message, items):

    if (len(items)) == 0:
        content = "There is nothing on the table."
    else:
        content = "There are "
        for i in range(len(items)):
            content = content + "[" + items[i] + "], "
        content = content + "on the table."
    prompt = {"role": "user", "content": content}
    message.append(prompt)

    return message

if __name__ == '__main__':

    model = YOLO("detect/train8/weights/best.pt")

    message = []
    prompt = {"role": "system", "content": "Answer questions to help user prepare all the required items for cooking. "
                                           "The required items are as follow: [1 Measure Cup], [1/2 Measure Cup], "
                                           "[1/4 Measure Cup], [Bowl], [Cork Hot Pad],"
                                           " [Metal Spoon], [Oatmeal], [Pan], [Salt], [Stirring Spoon], [Timer]. "
                                           "Other items are distractors."
                                           "You need remind user to prepare "
                                           "all of these items and let user remove distractors."}

    # [Bowl], [Measure Cups], [Oats], [Pan], [Salt container], [Spoon].
    message.append(prompt)
    prompt = {"role": "user", "content": "I going to cook oatmeal."}
    message.append(prompt)
    prompt = {"role": "system", "content": "Just for reminder, you need prepare [1 Measure Cup], "
                                           "[1/2 Measure Cup], [1/4 Measure Spoon], [Bowl], [Cork Hot Pad],"
                                           "[Metal Spoon], [Oatmeal], [Pan], [Salt], [Stirring Spoon], [Timer]."}
    message.append(prompt)

    pre_detected_items = []
    speech = gTTS(text=message[-1]["content"], lang='en', slow=False)
    speech.save("Voice_Prompt/speech.mp3")
    os.system("afplay Voice_Prompt/speech.mp3")
    while(True):
        time.sleep(5)
        s_time = time.time()
        success, img = cap.read()
        results = model(img, stream=True)
        detected_items = []
        for result in results:
            if result.boxes:
                for i in range(len(result.boxes)):
                    box = result.boxes[i]
                    if box.conf > 0.6:
                        class_id = int(box.cls)
                        object_name = model.names[class_id]
                        detected_items.append(object_name)

        detected_items = list(set(detected_items))
        print(detected_items)
        print("Time: ", time.time() - s_time)

        # if detected_items != pre_detected_items:
        #     pre_detected_items = detected_items
        #     continue
        # detected_items = ["Bowl", "Measure Cup", "Oats", "Pan", "Salt container", "Spoon", "Book", "Knife"]
        message = message_generate(message, detected_items)

        s_time = time.time()
        complements = openai.ChatCompletion.create(
            model="gpt-4",
            messages=message
        )
        result = complements
        message.append(result["choices"][0]["message"])
        print("Time: ", time.time() - s_time)

        speech = gTTS(text=result["choices"][0]["message"]["content"], lang='en', slow=False)
        speech.save("Voice_Prompt/speech.mp3")
        os.system("afplay Voice_Prompt/speech.mp3")

        if not os.path.exists('data'):
            os.makedirs('data')
        with open('data/LLM_4.json', 'w') as f:
            json.dump(message, f, indent=4)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()