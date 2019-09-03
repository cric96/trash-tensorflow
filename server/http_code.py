ok = 200 #after prediciton, a category is found, return a json object with this structure: {trashCategory: 'category'}
no_category = 201 # after prediction, no category is found
no_barcode = 202 #after barcode analysis, if no barcode is founder return this code