from rdflib import Graph
from pyshacl import validate

# Load your knowledge graph
kg = Graph()
kg.parse("hospital_1_LKG.ttl", format="turtle")

# Example SHACL validation
shacl_shape = """
@prefix clin: <http://clinicalkg.org/ontology#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .

clin:PatientShape a sh:NodeShape ;
    sh:targetClass clin:Patient ;
    sh:property [
        sh:path clin:hasDiagnosis ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] .
"""

conforms, results_graph, results_text = validate(kg, shacl_graph=shacl_shape)
print(f"Conforms: {conforms}\nResults: {results_text}")
