import requests
import os
import time


output_dir = "fake_images"
os.makedirs(output_dir, exist_ok=True)


num_images = 10000

for i in range(num_images):
    try:
        
        url = "https://thispersondoesnotexist.com"
        
       
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            
            with open(os.path.join(output_dir, f"fake_{i}.jpg"), "wb") as file:
                file.write(response.content)
            print(f"Image {i + 1} téléchargée.")
        
        
        time.sleep(1)
    except Exception as e:
        print(f"Erreur : {e}")

print("Téléchargement terminé !")
