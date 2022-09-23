import numpy
from openquake.hazardlib import nrml, sourceconverter
from openquake.hazardlib.source.complex_fault import ComplexFaultSource
from openquake.hazardlib.source.simple_fault import SimpleFaultSource
from openquake.hazardlib.source.area import AreaSource
from openquake.hazardlib.source.point import PointSource
from openquake.hazardlib.source.multi_point import MultiPointSource
from openquake.hazardlib.geo.surface import ComplexFaultSurface, SimpleFaultSurface
import numpy as np
import matplotlib.pyplot as plt
import pickle
from os import path
from shapely.geometry import Polygon, Point, MultiPoint

def cleaner_range(start, end, h):
    """ Returns array holding bin edges that doesn't contain floating point wander.

    Floating point wander can occur when repeatedly adding floating point numbers together. The errors propogate and become worse over the sum. This function generates the
    values on an integer grid and converts back to floating point numbers through multiplication.

     Args:
        start (float)
        end (float)
        h (float): magnitude spacing

    Returns:
        bin_edges (numpy.ndarray)
    """
    # convert to integers to prevent accumulating floating point errors
    const = 100000
    start = np.floor(const * start)
    end = np.floor(const * end)
    d = const * h
    return np.arange(start, end + d / 2, d) / const

def return_rates(src, MeshGrid):

    rates = []
    magnitudes = []
    cells = []
    src_type = []
    if isinstance(src, SimpleFaultSource):
        try:
            surf1 = SimpleFaultSurface.from_fault_data(src.fault_trace,
                                                      src.upper_seismogenic_depth, src.lower_seismogenic_depth,
                                                      src.dip, rupt_spacing)
            trace = src.fault_trace
            trace.flip()
            surf2 = SimpleFaultSurface.from_fault_data(trace,
                                                       src.upper_seismogenic_depth, src.lower_seismogenic_depth,
                                                       src.dip, rupt_spacing)
        except:
            try:
                surf1 = SimpleFaultSurface.from_fault_data(src.fault_trace,
                                                          src.upper_seismogenic_depth, src.lower_seismogenic_depth,
                                                          src.dip, 5)
                trace = src.fault_trace
                trace.flip()
                surf2 = SimpleFaultSurface.from_fault_data(src.fault_trace.flip(),
                                                           src.upper_seismogenic_depth, src.lower_seismogenic_depth,
                                                           src.dip, rupt_spacing)
            except:
                pass


        mesh = surf1.mesh.array
        n_points = mesh.shape[1] * mesh.shape[2]
        idx = np.argwhere(mesh[2, :, :] <= max_depth)
        new_points1 = surf1.mesh.array[:, idx[:, 0], idx[:,1]]

        mesh2 = surf2.mesh.array
        n_points2 = mesh2.shape[1] * mesh2.shape[2]
        idx = np.argwhere(mesh2[2, :, :] <= max_depth)
        new_points2 = surf2.mesh.array[:, idx[:, 0], idx[:,1]]
        new_points2 = new_points1[new_points2[:,2] != 0]
        new_points = np.vstack((new_points1, new_points2))


        new_n_points = new_points.shape[1]




        ratio = new_points.shape[1] / n_points
        rates_m = []
        mws = []
        for i, j in src.get_annual_occurrence_rates():
            mws.append(i)
            rates_m.append(j * ratio)

        for pp in new_points.T:
            cells.append([pp[0], pp[1]])
            magnitudes.append(mws)
            rates.append(np.array(rates_m) / new_n_points)
            src_type.append('simple_fault')

    elif isinstance(src, ComplexFaultSource):
        surf = ComplexFaultSurface.from_fault_data(src.edges, rupt_spacing)
        mesh = surf.mesh.array

        n_points = mesh.shape[1] * mesh.shape[2]
        idx = np.argwhere(mesh[2, :, :] <= max_depth)
        new_points = surf.mesh.array[:, idx[:, 0], idx[:,1]]


        new_n_points = new_points.shape[1]
        ratio = new_points.shape[1] / n_points
        rates_m = []
        mws = []
        for i, j in src.get_annual_occurrence_rates():
            mws.append(i)
            rates_m.append(j * ratio)

        for pp in new_points.T:
            cells.append([pp[0], pp[1]])
            magnitudes.append(mws)
            rates.append(np.array(rates_m) / new_n_points)
            src_type.append('complex_fault')

    elif isinstance(src, AreaSource):
        poly = Polygon([(i,j) for i,j in zip(src.polygon.lons, src.polygon.lats)])
        points = poly.intersection(MeshGrid)
        # print(points)
        try:
            xy = np.array([i.coords.xy for i in points]).squeeze()
        except TypeError:
            xy = np.array([points.coords.xy]).squeeze().reshape((-1, 2))
        counts = src.get_annual_occurrence_rates()
        rates = []
        magnitudes = []

        mws = []
        rates_m = []
        for i, j in counts:
            mws.append(i)
            rates_m.append(j)

        for k in xy:
            if len(k) > 0:
                cells.append([k[0], k[1]])
                magnitudes.append(mws)
                rates.append(np.array(rates_m)/xy.shape[0])
                src_type.append('area_source')


    elif isinstance(src, PointSource):

        rates_m = []
        mws = []
        for i, j in src.get_annual_occurrence_rates():
            mws.append(i)
            rates_m.append(j)
        cells.append([src.location.longitude, src.location.latitude])
        magnitudes.append(mws)
        rates.append(np.array(rates_m))
        src_type.append('point_source')

    elif isinstance(src, MultiPointSource):

        for PointSrc in src:

            rates_m = []
            mws = []
            for i, j in PointSrc.get_annual_occurrence_rates():
                mws.append(i)
                rates_m.append(j)
            cells.append([PointSrc.location.longitude, PointSrc.location.latitude])
            magnitudes.append(mws)
            rates.append(np.array(rates_m))
            src_type.append('multipoint_source')

    else:
        print('another', src.__class__)

    return cells,  magnitudes, rates , src_type

import geopandas as gpd
import pandas as pd
def project2mesh(mesh_fn, Cells, Magnitudes, Rates, SrcType):
    mm = []
    for i in Magnitudes:
        mm.extend(i)
    min_mag = np.round(min(mm),1)
    max_mag = np.round(max(mm),1)
    dm = 0.2

    mesh = np.genfromtxt(mesh_fn, delimiter=',')
    magnitudes = cleaner_range(min_mag, max_mag, dm).round(1)
    data = np.zeros((mesh.shape[0], 6 + len(magnitudes)))


    data[:, 0] = np.round(mesh[:, 0] - 0.05, 1)
    data[:, 1] = np.round(mesh[:, 0] + 0.05, 1)
    data[:, 2] = np.round(mesh[:, 1] - 0.05, 1)
    data[:, 3] = np.round(mesh[:, 1] + 0.05, 1)
    data[:, 4] = np.zeros(mesh.shape[0])
    data[:, 5] = 30*np.ones(mesh.shape[0])
    df = pd.DataFrame({'lon': Cells[:, 0], 'lat':Cells[:, 1]})
    df['coords'] = list(zip(df['lon'], df['lat']))
    df['coords'] = df['coords'].apply(Point)
    points = gpd.GeoDataFrame(df, geometry='coords')
    polygons = []
    for i in data:
        polygons.append(Polygon([(i[0], i[2]), (i[1]-0.0000001, i[2]),
                                 (i[1]-0.0000001, i[3]-0.0000001), (i[0], i[3]-0.0000001)]))

    grid = gpd.GeoDataFrame({'geometry': polygons})
    points_in_poly = gpd.tools.sjoin(points, grid, predicate='within', how='left')
    idxs = points_in_poly.index_right.to_numpy()

    for n, (m, r) in enumerate(zip(Magnitudes, Rates)):
        if n % 5000 == 0:
            print('proj', n)
        try:
            m_ind = np.argmin(np.abs(magnitudes + 0.001 - m[0]))
            data[int(idxs[n]), 6 + m_ind: 6 + m_ind + len(r)] += r
        except:
            try:
                dmm = np.diff(m)[0]
                dm_delta = int(np.round(dm/dmm))
                ratess = r[::dm_delta]
                mmms = m[::dm_delta]
                m_ind = np.argmin(np.abs(magnitudes + 0.001 - mmms[0]))
                data[int(idxs[n]), 6 + m_ind: 6 + m_ind + len(ratess)] += ratess
            except:
                # print('uh')
                continue
    # zero_ind = np.argwhere(data[:,  6:].sum(axis=1) == 0 )
    # for i in zero_ind:
    #     try:
    #         data[i, 6:] = data[i+1, 6:]
    #     except:
    #         continue
    return data


if __name__ == '__main__':
    subd= ['interface_v12b/CaA_IF2222222_M40.xml',
    'interface_v12b/CyA_IF2222222_M40.xml',
    'interface_v12b/GiA_IF2222222_M40.xml',
    'interface_v12b/HeA_IF2222222_M40.xml']

    # files = ['seifa_model.xml', 'area_source_model.xml', 'faults_backg_source_model.xml']
    files = [['ssm_v09/seis_ver12b_fMthr_asm_ver12e_winGT_fs017_agbrs_point.xml', 'fsm_v09/fs_ver09e_model_aGR_SRA_MA_fMthr.xml'],
             ['ssm_v09/seis_ver12b_fMthr_asm_ver12e_winGT_fs017_agbrs_point.xml', 'fsm_v09/fs_ver09e_model_aGR_SRA_ML_fMthr.xml'],
             ['ssm_v09/seis_ver12b_fMthr_asm_ver12e_winGT_fs017_agbrs_point.xml', 'fsm_v09/fs_ver09e_model_aGR_SRA_MU_fMthr.xml'],
             ['ssm_v09/seis_ver12b_fMthr_asm_ver12e_winGT_fs017_agbrs_point.xml', 'fsm_v09/fs_ver09e_model_aGR_SRL_MA_fMthr.xml'],
             ['ssm_v09/seis_ver12b_fMthr_asm_ver12e_winGT_fs017_agbrs_point.xml', 'fsm_v09/fs_ver09e_model_aGR_SRL_ML_fMthr.xml'],
             ['ssm_v09/seis_ver12b_fMthr_asm_ver12e_winGT_fs017_agbrs_point.xml', 'fsm_v09/fs_ver09e_model_aGR_SRL_MU_fMthr.xml'],
             ['ssm_v09/seis_ver12b_fMthr_asm_ver12e_winGT_fs017_agbrs_point.xml', 'fsm_v09/fs_ver09e_model_aGR_SRU_MA_fMthr.xml'],
             ['ssm_v09/seis_ver12b_fMthr_asm_ver12e_winGT_fs017_agbrs_point.xml', 'fsm_v09/fs_ver09e_model_aGR_SRU_ML_fMthr.xml'],
             ['ssm_v09/seis_ver12b_fMthr_asm_ver12e_winGT_fs017_agbrs_point.xml', 'fsm_v09/fs_ver09e_model_aGR_SRU_MU_fMthr.xml'],
             ]
    # files = [['ssm_v09/seis_ver12b_fMthr_asm_ver12e_winGT_fs017_agbrs_point.xml', 'fsm_v09/fs_ver09e_model_aGR_SRA_MA_fMthr.xml']  ]
    # files = ['asm_v12e/asm_ver12e_winGT_fs017_hi_abgrs_maxmag_low.xml',
    #          'asm_v12e/asm_ver12e_winGT_fs017_hi_abgrs_maxmag_mid.xml',
    #          'asm_v12e/asm_ver12e_winGT_fs017_hi_abgrs_maxmag_upp.xml',
    #          'asm_v12e/asm_ver12e_winGT_fs017_lo_abgrs_maxmag_low.xml',
    #          'asm_v12e/asm_ver12e_winGT_fs017_lo_abgrs_maxmag_mid.xml',
    #          'asm_v12e/asm_ver12e_winGT_fs017_lo_abgrs_maxmag_upp.xml',
    #          'asm_v12e/asm_ver12e_winGT_fs017_mid_abgrs_maxmag_low.xml',
    #          'asm_v12e/asm_ver12e_winGT_fs017_mid_abgrs_maxmag_mid.xml',
    #          'asm_v12e/asm_ver12e_winGT_fs017_mid_abgrs_maxmag_upp.xml',
    #          'asm_v12e/asm_ver12e_winGT_fs017_pareto_abgrs_cornermag_low.xml',
    #          'asm_v12e/asm_ver12e_winGT_fs017_pareto_abgrs_cornermag_mid.xml',
    #          'asm_v12e/asm_ver12e_winGT_fs017_pareto_abgrs_cornermag_upp.xml'
    #          ]
    mesh_fn = 'mesh_europe.csv'
    mesh = np.genfromtxt(mesh_fn, delimiter=',')
    mesh_shp = MultiPoint([Point(i[0],i[1]) for i in mesh])
    rupt_spacing = 3
    area_mesh = 8
    max_depth = 35
    reload = False
    mw_bin = 0.1
    for filename in files:

        if reload is False:

            parser = sourceconverter.SourceConverter(area_source_discretization=area_mesh,
                                                     width_of_mfd_bin=mw_bin)
            if isinstance(filename, list):
                source_list = filename
                filename = filename[-1]
                source_list.extend(subd)
                a = nrml.read_source_models(source_list, parser)
            else:
                source_list = [filename, 'asm_v12e/asm_ver12e_winGT_fs017_twingr.xml']
                source_list.extend(subd)
                a = nrml.read_source_models(source_list, parser)
            src_models = []
            for i in a:
                src_models.append(i)
            Rates = []
            Magnitudes = []
            Cells = []
            SrcType = []
            srcs = []

            for smm in src_models:
                for sgc in smm:
                    for src in sgc:
                        srcs.append(src)
            for i, src in enumerate(srcs):
                if i % 50 == 0:
                    print('src', i)
                cc,mm, rr, st= return_rates(src, mesh_shp)
                if len(cc) > 0:
                    Cells.extend(cc)
                    Magnitudes.extend(mm)
                    Rates.extend(rr)
                    SrcType.extend(st)
                else:
                    print('ho')
            with open(path.splitext(filename)[0] + '.obj', 'wb') as pick:
                pickle.dump((Cells, Magnitudes, Rates, SrcType), pick)
        else:
            if isinstance(filename, list):
                filename = filename[-1]
            with open(path.splitext(filename)[0] + '.obj', 'rb') as pick:
                Cells, Magnitudes, Rates, SrcType = pickle.load( pick)

        np.savetxt(path.splitext(filename)[0] + '_original.csv', np.array(Cells), delimiter=',')


        plt.scatter([i[0] for i in Cells], [i[1] for i in Cells],
                     c = np.log10([np.sum(i) for i in Rates]), vmin=-8, vmax=-3, s=0.01)
        plt.show()

        data = project2mesh(mesh_fn, np.array(Cells), Magnitudes, Rates, SrcType)
        plt.scatter((data[:, 0]+data[:, 1])/2., (data[:, 2]+data[:, 3])/2.,
                    c=np.log10(data[:, 6:].sum(axis=1)), s=0.04)
        plt.show()
        mm = []
        for i in Magnitudes:
            mm.extend(i)
        min_mag = min(mm)
        max_mag = max(mm)
        dm = 0.2
        magnitudes = cleaner_range(min_mag, max_mag, dm).round(1)

        if min(magnitudes) == 4.6:
            magnitudes += 0.1
        elif max(magnitudes) == 4.5:
            magnitudes += 0.2
        elif max(magnitudes) == 4.55:
            magnitudes += 0.15

        header = 'lon_min,lon_max,lat_min,lat_max,depth_min,depth_max,' + ','.join([f'{i :.1f}' for i in magnitudes[:data.shape[1] -6]])
        fn_csv = path.splitext(filename)[0] + '.csv'
        np.savetxt(f'{fn_csv}', data, delimiter=',', header=header)

