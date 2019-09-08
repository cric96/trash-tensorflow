
packagings_to_category = {
    "Indifferenziato" : [ "poliaccoppiati", "tetrapak", "tetrapack","composite-packaging", "matériau-composite"],
    "Carta" : ["car", "pap","charte"],
    "Plastica" : ["pet", "plast", "kunstsoff", "pvc","pe", "pctfe","pvdc" ],
    "Alluminio" : ["lattine", "alluminio", "metallo", "acciaio","alumin", "métal"],
    "Vetro" : ["vetro", "verre", "glas"],
}
separator = ""
#take a string contains a set of packaging tags and try to find the trash category, the string must be lower case
def find_category_from_packaging(packaging_string):
    for category, packagings in packagings_to_category.items(): #iterate through the map 
        for packaging in packagings: #iterate through packaging tags defined in map
            if packaging in packaging_string: #verify if there is a match with the packaging string passed
                return category
    return ""
def get_trash_category(product):
    result = ""
    if "packaging_tags" in product:
        print(product["packaging_tags"])
        #if there isn't match with packaging, try to find into packaging_tags
        #packaging tags is an list, to convert into string you need to use join.
        result = find_category_from_packaging(separator.join(product["packaging_tags"]).lower()) 
    if result == "" and "packaging" in product:
        print(product["packaging"])
        #check if the category could be find in packaging attribute
        result = find_category_from_packaging(product["packaging"].lower())
    return result  
    