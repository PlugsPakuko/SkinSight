from ultralytics import YOLO
# Run inference
def predict(INPUT, MODEL):
    if(MODEL == 'abrasion'):
        results = abrasion_model(INPUT)
    if(MODEL == 'burn'):
        results = burn_model(INPUT)
    if(MODEL == 'cut'):
        results = cut_model(INPUT)
    return results

# load model
burn_model = YOLO('./weights/SkinBurn.pt')
abrasion_model = YOLO('./weights/SkinBurn.pt')
cut_model = YOLO('./weights/SkinBurn.pt')