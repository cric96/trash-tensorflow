http_ok = 200
status_ok = 0
status_category_not_found = 1
status_barcode_not_found = 2
def category_found(category):
    return {"status" : status_ok, "category" : category} , http_ok
def category_not_found():
    return {"status" : status_category_not_found}, http_ok
def barcode_not_found():
    return {"status" : status_barcode_not_found}, http_ok