# myapp/views.py
from django.shortcuts import render
from .forms import ImageUploadForm
from .utils import predict_cat_or_dog

def classify_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            prediction = predict_cat_or_dog(image)
            return render(request, 'result.html', {'prediction': prediction})
    else:
        form = ImageUploadForm()

    return render(request, 'upload.html', {'form': form})
