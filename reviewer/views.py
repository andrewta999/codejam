from django.shortcuts import render
from .forms import Review
from . import test1
from . import summary

def test(request):
	result = None
	if request.method == 'POST':
		form = Review(request.POST)
		if form.is_valid():
			result = form.cleaned_data['Review']
			result = test1.get_kws(result, limit=6)
			result = ','.join(result)
	else:
		form = Review()
	return render(request, 'index.html', {'form': form, 'result': result})

# def summary(request):
# 	if request.method == 'POST':
# 		form = Review(request.POST)
# 		if form.is_valid():
