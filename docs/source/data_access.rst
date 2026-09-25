data_access.py (Legacy)
-----------------------

.. deprecated::
   This module (``esp_lab.data_access``) is the original monolithic data accessor
   inherited from the pre-refactor ``SMYLEutils`` / ``esp-tools`` era.
   It is retained for backward compatibility only.

   For new workflows, use the purpose-specific accessors instead:

   - **E3SM hindcast data**: :mod:`esp_lab.data_access_e3sm`
   - **CESM-SMYLE hindcast data**: :mod:`esp_lab.data_access_cesm_smyle`
   - **Observational references**: :mod:`esp_lab.data_access_obs`
   - **NMME model data**: :mod:`esp_lab.data_access_nmme`

.. automodule:: esp_lab.data_access
    :members:
    :undoc-members:
