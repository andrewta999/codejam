from django import forms

class Review(forms.Form):
	Review = forms.CharField(max_length=1000, widget=forms.Textarea())


