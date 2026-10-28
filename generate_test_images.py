import torch
from torchvision import datasets
from PIL import Image, ImageOps

def genereaza_imagini_test():
    print("Incarcam datele pentru a extrage cifre reale scrise de mana...")

    dataset = datasets.MNIST('./data', train=False, download=True)

    cifre_gasite = set()
    
    for img, label in dataset:
        if label not in cifre_gasite:
         
            img_inversata = ImageOps.invert(img)
            
            
            img_mare = img_inversata.resize((150, 150), Image.Resampling.LANCZOS)
            

            canvas = Image.new('L', (300, 300), color='white')
            canvas.paste(img_mare, (70, 50))
            
       
            nume_fisier = f"test_cifra_{label}.png"
            canvas.save(nume_fisier)
            print(f"Generat cu succes: {nume_fisier}")
            
            cifre_gasite.add(label)
            
        if len(cifre_gasite) == 10:
            break
            
    print("\nAi acum 10 imagini noi de test in folderul proiectului!")

if __name__ == "__main__":
    genereaza_imagini_test()