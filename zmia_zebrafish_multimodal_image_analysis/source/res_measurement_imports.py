import os.path

import skimage.io

from prairie_view_imports import *


def main():
    config = set_config(os.path.join('..',
                                     'configs',
                                     '2p_res_measure_v2.yml'))

    all_datasets = load_all_datasets(do_fail=True,
                                     lazy=False,
                                     ignore_cache=True)

    for d in all_datasets:
        # print('Voxel Pitch for dataset           "%27s" is %40s'
        #       % (d.name, d.voxel_pitch_um))
        # print('Laser Power (pockels) for dataset "%27s" is %40.3f'
        #       % (d.name, d.scan_metadata['laserPower'][0]))
        # print('Laser Wavelength for dataset      "%27s" is %40d'
        #       % (d.name, d.scan_metadata['laserWavelength'][0]))

        print(f'dataset info: {d.name:26s}, '
              f'{d.scan_metadata[pvmk.OBJECTIVE_LENS]:20s}, '
              f'{d.scan_metadata[pvmk.LASER_WAVELENGTH][0]:5d} nm, '
              f'{d.scan_metadata[pvmk.LASER_POWER][0]:5.1f} pockels, '
              f'{d.scan_metadata[pvmk.LASER_ATTENUATION][0]:4.1f} %, ')

        export_dir = os.path.join(config['data_directory'], 'exports')
        csutils.touchdir(export_dir)

        export_name = os.path.join(export_dir, f'{d.name}.tiff')
        skimage.io.imsave(export_name,
                          d.I,
                          plugin='tifffile',
                          imagej=True,
                          check_contrast=False)


if __name__ == '__main__':
    main()
