# ceoiler_portfolio

Goal for IR Project

Similar to HW2 where we use TFIDF to match queries and text. 
Need to add a bunch of images to this project to so we can use an image to text model to generate captions for each image
Models I found are CLIP and BLIP. Both can be used from the hugging face database and look pretty easy to implement
BLIP is more accurate but could take a little longer to process requests.
In additon I need to add a search bar into the UI of the homescreen and probably a new page that popualtes the images once we retreive them.

Goal is to classify images in a more general sense. My portfolio has photos ranging from sports photos (Lacrosse, volleyball, soccer, football, track, XC, etc), 
grad photos and other individual photo shoots, and then bunch of nature shots so I would ideally like to classify based off things like sunset, type of sport, features like water, lakes, rivers, beaches, trees, rocks, etc. Much mroe general than a specfic classificition. 

BLIP: https://huggingface.co/Salesforce/blip-image-captioning-base

CLIP: https://huggingface.co/docs/transformers/model_doc/clip