# ZMIA - Zebrafish Multi-modal Image Analysis

This repository is a collection of programs for the import, 
processing and analysis of larval zebrafish brain imaging data.

The primary goal of this work is toward the spatial registration of 
structural and functional brain imaging data.

### `README` Contents:
1. [Gettings Started](#getting-started)
2. [Configuration Files](#configuration-files)
3. [Shared Datasets](#shared-datasets)

## Getting Started

The easiest way to start will be to clone this code repository, and use conda 
to setup a python environment.You can do this via the following steps:

1. [Clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) 
   this repository into a folder (e.g. `.../zmia/`) on your computer.
   
2. After cloning this repository, you will need to then import its 
   submodules via the following command:
   ```bash
   git submodule update --init --recursive
   ```
     
3. Download and install a conda-based package management software if you do not already have it. I recommend Mambaforge for faster enviornment builds.
   * [Mambaforge (recommended)](https://github.com/conda-forge/miniforge#mambaforge)
   * [Miniconda](https://github.com/conda-forge/miniforge#miniforge3)
   * [Anaconda](https://www.anaconda.com/products/individual)

4. **Environment Creation:** From the shell (look for ___conda prompt on Windows) `cd` to the cloned repository directory, then run 
   the following command:
   ```bash
   conda env create --prefix ./envs/zmia-env --file ./envs/zmia-env-3.9.yml
   ```
   *Note: you may replace `zmia-env-3.9.yml` with any of the enviornment files in the [`envs` directory](/envs)
   based on what you may be trying to run.*

5. Activate the anaconda environment by the following command
   ```bash
   conda activate ./envs/zmia-env
   ```
   From here you have completed the setup to run this repository's code.
6. **Viewing Shared Data (optional)**: If you want to view a shared dataset:
   1. First, update the configuration `.yaml` file for that dataset, such that the `data_directory` and 
      `output_directory` point to the correct location on your file system.
   2. Then run the jobfile associated with that shared dataset. For example to view the CS-II-14 
      shared data you would run the following command (with zmia-env activated):
      ```bash
      python ./source/jobs/local_cs-ii-14_shared-data-setup.py
      ``` 
      
   Please see the [Shared Datasets](#shared-datasets) section for more information.
  

## Configuration Files
The definition of datasets should be done via configuration files. These 
are [`.yml`](https://yaml.org/) files with information defining where to 
find and store data on your filesystem. Here is an example configuration:

```yaml
# my_config.yml
data_directory: '/mnt/c/data'
log_directory: '/mnt/c/user/logs'
datasets:
  - name: 'GCaMP_TZ (512x512)'
    path: 'test_data/dataset_01'

  - name: 'GCaMP_Z (2048x2048)'
    path: 'test_data/dataset_02'
```

In brief, the configuration file should be structured as follows:
* `data_directory` **this key is required**, it should map to a string scalar which is the
  top directory containing all data for the dataset. *For robustness make
  this an absolute path.*
* `log_directory` key should map to a string scalar which is 
  the directory to send log files.
* `datasets` key should map to a sequence of mappings in which each entry can 
  have the following keys.
  * `path` this key is **required**, it should map to a string scalar which 
    points to the directory containing the dataset's images and PrairieView 
    `.xml` file (or a `PVDataset` `.h5` file).
  * `name` this key, if included shoudl map to a string scalar name which will be 
     added to the dataset's metadata.

The order of datasets in the configuration file is important, each dataset will
be referenced by its index, beginning with zero. For example, in 
`my_config.yml` dataset `"GCaMP_Z (2048x2048)"` has index `1`.

### Configuration Files and Git
Use the following command to stop tracking changes to a config file
(e.g. for a template config which will get updated after committing):
```bash
git update-index --assume-unchanged ./configs/{config-name}.yml
```
Begin tracking again with:
```bash
git update-index --no-assume-unchanged ./configs/{config-name}.yml
```

## Shared Datasets
*Please email me at c_swain at mit dot edu to be added to any shared folders.*

### 1. Zebrafish 2-photon Imaging Data (id: `22-03-14_live_imaging_session (CS-I-90-A)`)
* [Dropbox Shared Folder](https://www.dropbox.com/sh/bxq9r8k2z3nc0o9/AABG2BzCZE2ZD_aTg9VcK5ita?dl=0)
* Datasets
  * `./data_for_fiji/`
    * high resolution 2p z-stack
      * combined tiff file
    * low resolutin 2p t-z-stack
      * combined tiff file
  * `./data_for_python/`
    * high resolution 2p z-stack      
      * raw prairie view image data
      * raw prairie view xml file
      * *this dataset can be read in by the python script*
  * `./corban_220314_dataset.yml` configuration file

### 2. Zebrafish Multi-modal Imaging Data (id `cs-ii-14_corban-swain-shared-zmia-data`)
* Locations
  * [Dropbox Shared Zipfile and Tiff Files](https://www.dropbox.com/sh/mo0o9mtrd9u6xwi/AADuE_Jg9_F-tfXMMzhMzqGua?dl=0)
    * Unzip Zipfile with [7zip](https://www.7-zip.org/) 
    * `tiff_files` directory contains tiffs for viewing data in ImageJ
  * BoydenLab : openmind.mit.edu : NESE storage
    * `.../c_swain/zf_correlative_microscopy/shared_data/cs-ii-14_shared-data`
  * BoydenLab : Portable Drive
    * email c_swain at mit dot edu
* Dataset Information: [See Configuration File](/configs/cs-ii-14_shared-config.yml)
  * Live 2 photon Calcium Imaging
  * Live Behavioral (tail movement) imaging
  * *In vivo* GCaMP Morphological Reference
  * *Ex vivo* GCaMP Immunostained
* Configuration file will need to be updated to point to the 
  `zf_correlative_microscopy` directory within `cs-ii-14_shared-data`
* **View this dataset by running the associated script** [`local_cs-ii-14_shared-data-setup.py`](/source/jobs/local_cs-ii-14_shared-data-setup.py)
  1. First activate the zmia-env as described in ["Getting Started"](#getting-started)
  1. Then run the following command from the repository directory:
  ```bash
  python ./source/jobs/local_cs-ii-14_shared-data-setup.py
  ```  


## Dataset Naming and Directory Structuring

### General Notes
* When possible, use only lowercase letters.
* When possible, do not include spaces and use underscores instead. 
* Keep data and large data files out of the code repositories.
* Do not reveal directory structure within the repositories within code, 
  reference a private text, `.yaml`, or other file flagged to be ignored by git.

### Directory Structure
*Note*: files should only appear at the bolded locations marked "**FILE(S):**"

* *some drive`/`*
  * `{username}/`
    * `{project_name}/`
      * `raw_data/` (*Use this directory only for directly acquired data, not
         simulated or externally acquired data. Files in this directory should NOT be modified or deleted.*) 
        * `{notebook_id}_{acquisition_mode}_{acquisition_date}/`
          * `{fish_id}/`
            * **FILES: raw data files**
          * **FILE: `acquisition_notes.txt`** (*Use this file to note down observations or comments regarding 
            the acquisition session.*)
      * `analyzed_and_generated_data/`
        * `{notebook_id}_{fish_id}_{creation_date}/`
          * **FILES: analyzed and generated data files**
      * `external_data/` (*Use this directory for data brought in from external 
         sources like collaborators or public datasets.*)
        * `{external_dataset_name}_{external_dataset_date}/`
          * **FILES: external dataset files**
      * `repos/` (*Use this directory for code repositories.*)
        * `{repository_name}/`
          * **FILE: `readme.md`** (*All code repositories (and subfolders too) should have a descriptive readme.*)
          * `envs/` (*Specifications fot the enviornments to run the code*)
          * `source/` (*Source code; root directory for running repository programs*)
            * **FILES: source code files**
          * **FILES: other repository files**
      * `logs/`
        * `{log_repository_name}/`
          * **FILES: `{logfile_date}-{logfile_time}_{logfile_name}.log`**
      * `misc/` (*Use this directory for any unsorted or miscellaneous files, nothing "mission critical."*)

### Naming Convention Wildcards

#### `{username}`
* Can be anything, use mit kerberos when possible.
* Example:
  * `c_swain`

#### `{project_name}`
* Name of the project.
* May be one of:
  * `zf_correlative_microscopy`
  * `zf_olfactory_stimulator_development`
  * `lightfield_multiview_reconstruction`
  * `other_files` (*For files not belonging to a particular project.*) 

#### `{notebook_id}`
* Identification of notebook page (reference to physical or Evernote lab 
  notebook)
* Formats:
  * `{initials}-{book number as roman numeral}-{page or entry number}`
  * If the notebook ref is not applicable or unknown use `no-nb-ref`
* Examples:
  * `cs-ii-04`
  * `no-nb-ref`

#### `{aquisition_mode}`
* Method used to obtain the raw data.
* Can be one of:
  * `confocal`
  * `2-photon`
  * `wide-field-camera`
  * `behavioral-camera`
  * `sensor-outputs`

#### Dates 
* Includes: 
  * `aquisition_date`, the date the dataset was acquired
  * `creation_date`, the date the analysis directory was first created 
  * `external_dataset_date`, the date the external dataset was published or shared
  * `logfile_date`, the date the logfile was created
* A calendar date reference
* Format: 
  * `YYYYMMDD`
    * If a value is unknown replace it with zeros
* Examples:
  * `20220721`
  * `20220000`
  * `20220500`
  * `00000000`

#### `{fish_id}`
* Identification for the fish larva(e). Might correspond to a single fish or a
  collection of fish with the same stimulation and/or processing conditions.
* Can be anything, usually a single letter.
* Use `unknown-fish` if fish id unknown.
* Use `no-fish` if no fish is referenced for the data.
* Examples:
  * `A`
  * `B3`
  * `unknown-fish`
  * `no-fish`

#### `{external_dataset_name}`
* The name of the external dataset, can be anything
* Examples:
  * `z_brain_atlas`


#### `{*repository_name}`
* Includes:
  * `{repository_name}`, the name of the code repository, ideally matches the name on github
  * `{log_repository_name}`, the name of the code repository which produced the logs 
* Examples:
  * `c-swain-python-utils`
  * `zmia-zebrafish-multimodal-image-analysis`


