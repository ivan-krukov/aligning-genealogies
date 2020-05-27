

class Haplotype(object):

    """
    A Haplotype is an object that encapsulates the paternal/maternal
    copy of a DNA segment in a particular individual. An individual
    in a pedigree is composed of 2 Haplotypes.
    """

    def __init__(self, haplotype_id, individual_id):

        self.id = haplotype_id
        self.individual_id = individual_id

        self.parent_haplotype = None
