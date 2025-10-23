from rdflib import Graph, URIRef, RDF
from pyshacl import validate

def technical_validation(ttl_file):
    # Initialize
    kg = Graph()
    clin_ns = "http://clinicalkg.org/ontology#"
    
    # 1. Load with performance optimization
    kg.parse(ttl_file, format="turtle", publicID=clin_ns)
    kg.bind("clin", clin_ns)
    
    # 2. Define URIs
    Patient = URIRef(clin_ns + "Patient")
    hasDiagnosis = URIRef(clin_ns + "hasDiagnosis")
    hasFeature = URIRef(clin_ns + "hasFeature")

    # 3. Structural Validation
    patients = list(kg.subjects(RDF.type, Patient))
    print(f"STRUCTURAL: Found {len(patients)} patients")
    
    # 4. Functional Validation
    missing_diag = sum(1 for p in patients if not list(kg.objects(p, hasDiagnosis)))
    print(f"FUNCTIONAL: Missing diagnoses: {missing_diag}/{len(patients)}")

    # 5. Data Quality
    features = {f for p in patients for f in kg.objects(p, hasFeature)}
    print(f"QUALITY: Unique features: {len(features)}")

    # 6. SHACL Validation (Fixed with proper prefixes)
    shacl_rules = """
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix clin: <%s> .
    
    clin:PatientShape a sh:NodeShape ;
        sh:targetClass clin:Patient ;
        sh:property [
            sh:path clin:hasDiagnosis ;
            sh:minCount 1 ;
        ] .
    """ % clin_ns
    
    conforms, _, _ = validate(kg, shacl_graph=shacl_rules)
    print(f"SHACL: Validation passed: {conforms}")

if __name__ == "__main__":
    technical_validation("hospital_1_LKG.ttl")
