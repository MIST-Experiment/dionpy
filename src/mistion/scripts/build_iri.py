def build_iri():
    from datetime import datetime
    import iri2016
    iri2016.IRI(datetime.now, [1000, 1000, 1], 90, 90)