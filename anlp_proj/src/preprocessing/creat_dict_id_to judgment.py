import json
import xml.etree.ElementTree as ET


def parse_xml(xml_file, result = {}):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    diseases = root.findall('diseases/disease')

    for disease in diseases:
        disease_name = disease.get('name')
        docs = disease.findall('doc')
        for doc in docs:
            doc_id = doc.get('id')
            judgment = doc.get('judgment')
            if doc_id not in result:
                result[doc_id] = []
            if judgment == 'Y':
                result[doc_id].append(disease_name)
    return result

xml_file_path = 'obs_del/obesity_standoff_annotations_test_intuitive.xml'  # Replace with your XML file path

parsed_data = parse_xml(xml_file_path)
with open('obs_id_to_disease_test_intuitive.json', 'w') as file:
    json.dump(parsed_data, file)
print(parsed_data)

xml_file_path = 'obs_del/obesity_standoff_annotations_test_textual.xml'  # Replace with your XML file path

parsed_data = parse_xml(xml_file_path)
with open('obs_id_to_disease_test_textual.json', 'w') as file:
    json.dump(parsed_data, file)
print(parsed_data)

xml_file_path = 'obs_del/obesity_standoff_intuitive_annotations_training.xml'  # Replace with your XML file path

parsed_data = parse_xml(xml_file_path)
xml_file_path = 'obs_del/obesity_standoff_intuitive_annotations_training.xml'  # Replace with your XML file path
xml_file_path2 = 'obs_del/obesity_standoff_annotations_training_addendum2.xml'

parsed_data = parse_xml(xml_file_path2, parsed_data)
with open('obs_id_to_disease_train_intuitive.json', 'w') as file:
    json.dump(parsed_data, file)

xml_file_path = 'obs_del/obesity_standoff_textual_annotations_training.xml'  # Replace with your XML file path
xml_file_path2 = 'obs_del/obesity_standoff_annotations_training_addendum.xml'
parsed_data = parse_xml(xml_file_path)
parsed_data = parse_xml(xml_file_path2, parsed_data)
with open('obs_id_to_disease_train_textual.json', 'w') as file:
    json.dump(parsed_data, file)

xml_file_path1 = 'obs_del/obesity_standoff_annotations_training_addendum3_intuitive.xml'
parsed_data = parse_xml(xml_file_path)
with open('obs_id_to_disease_train2_intuitive.json', 'w') as file:
    json.dump(parsed_data, file)

xml_file_path1 = 'obs_del/obesity_standoff_annotations_training_addendum3_textual.xml'
parsed_data = parse_xml(xml_file_path)
with open('obs_id_to_disease_train2_textual.json', 'w') as file:
    json.dump(parsed_data, file)

