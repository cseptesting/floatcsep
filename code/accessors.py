def query_gcmt(*args, **kwargs):
    """ Return gCMT catalog from ISC online web-portal

        Args: 
            (follow csep.query_comcat for guidance)

        Returns:
            out (csep.core.catalogs.AbstractBaseCatalog): gCMT catalog
    """
    pass

def download_from_zenodo(resource_id):
    """ Downloads file from Zenodo and returns checksum for each file

        Note: Returns here should be a list even if one object is found

        Args:
            resource_id (int): resource id from zenodo

        Returns:
            checksum (string): md5 checksum from model
    """
    pass

