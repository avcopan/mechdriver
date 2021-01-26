""" es_runners for initial geometry optimization
"""

import shutil
import numpy
import automol
import elstruct
import autofile
# from phydat import instab_fgrps
from phydat import phycon
from mechroutines.es import runner as es_runner
from mechlib import structure
from mechlib import filesys
from mechlib.submission import qchem_params
from mechlib.amech_io import printer as ioprinter


def reference_geometry(spc_dct_i, spc_info,
                       mod_thy_info,
                       thy_run_fs, thy_save_fs,
                       cnf_save_fs,
                       ini_thy_save_path, mod_ini_thy_info,
                       run_fs,
                       opt_script_str, overwrite,
                       kickoff_size=0.1, kickoff_backward=False,
                       **opt_kwargs):
    """ determine what to use as the reference geometry for all future runs
    If ini_thy_info refers to geometry dictionary then use that,
    otherwise values are from a hierarchy of:
    running level of theory, input level of theory, inchis.
    From the hierarchy an optimization is performed followed by a check for
    an imaginary frequency and then a conformer file system is set up.
    """

    # Initialize empty return
    ret = None

    if run_fs[0].file.info.exists([]):
        inf_obj = run_fs[0].file.info.read([])
        if inf_obj.status == autofile.schema.RunStatus.RUNNING:
            ioprinter.already_running('Reference geometry', run_fs[0].path([]))
            return ret
    else:
        [prog, method, basis, _] = mod_thy_info
        status = autofile.schema.RunStatus.RUNNING
        inf_obj = autofile.schema.info_objects.run(
            job='', prog=prog, version='version', method=method, basis=basis,
            status=status)
        run_fs[0].file.info.write(inf_obj, [])

    exists = thy_save_fs[-1].file.geometry.exists(mod_thy_info[1:4])
    if not exists:
        ioprinter.info_message(
            'No energy found in save filesys. Running energy...')
        _run = True
    elif overwrite:
        ioprinter.info_message(
            'User specified to overwrite energy with new run...')
        _run = True
    else:
        _run = False

    if _run:
        ioprinter.info_message('Obtaining some initial guess geometry.')
        geo_init = _obtain_ini_geom(spc_dct_i,
                                    ini_thy_save_path, mod_ini_thy_info,
                                    overwrite)

        if geo_init is not None:
            ioprinter.debug_message(
                'Assessing if there are any functional groups',
                'that cause instability')
            ioprinter.debug_message('geo str\n', automol.geom.string(geo_init))
            if True: #_functional_groups_stable(geo_init, thy_save_fs, mod_thy_info):
                zma_init = automol.geom.zmatrix(geo_init)
                if not automol.geom.is_atom(geo_init):
                    geo_found = _optimize_molecule(
                        spc_info, zma_init,
                        mod_thy_info, thy_run_fs, thy_save_fs,
                        cnf_save_fs,
                        run_fs,
                        opt_script_str, overwrite,
                        kickoff_size=kickoff_size,
                        kickoff_backward=kickoff_backward,
                        **opt_kwargs)
                else:
                    geo_found = _optimize_atom(
                        spc_info, zma_init,
                        mod_thy_info, thy_run_fs,
                        cnf_save_fs, run_fs,
                        overwrite, opt_script_str, **opt_kwargs)
            else:
                geo_found = True
                ioprinter.info_message(
                    'Found functional groups that cause instabilities')
        else:
            geo_found = False
            ioprinter.warning_message(
                'Unable to obtain an initial guess geometry')
    else:
        geo_found = True
        thy_path = thy_save_fs[-1].path(mod_thy_info[1:4])
        ioprinter.existing_path('Initial geometry', thy_path)

    # Write the job status into the run filesystem
    if geo_found:
        inf_obj.status = autofile.schema.RunStatus.SUCCESS
        run_fs[0].file.info.write(inf_obj, [])
    else:
        inf_obj.status = autofile.schema.RunStatus.FAILURE
        run_fs[0].file.info.write(inf_obj, [])

    return geo_found


def _obtain_ini_geom(spc_dct_i, ini_thy_save_path,
                     mod_ini_thy_info, overwrite):
    """ Obtain an initial geometry to be optimized. Checks a hieratchy
        of places to obtain the initial geom.
            (1) Geom dict which is the input from the user
            (2) Geom from inchi
    """

    geo_init = None

    # Obtain geom from thy fs or remove the conformer filesystem if needed
    if not overwrite:
        ini_cnf_save_fs = autofile.fs.conformer(ini_thy_save_path)
        ini_min_cnf_locs, _ = filesys.mincnf.min_energy_conformer_locators(
            ini_cnf_save_fs, mod_ini_thy_info)
        if ini_min_cnf_locs:
            geo_init = ini_cnf_save_fs[-1].file.geometry.read(ini_min_cnf_locs)
            ioprinter.info_message(
                'Getting inital geometry from inplvl at path',
                '{}'.format(ini_cnf_save_fs[-1].path(ini_min_cnf_locs)))
    else:
        ioprinter.debug_message(
            'Removing original conformer save data for instability')
        ini_cnf_save_fs = autofile.fs.conformer(ini_thy_save_path)
        for locs in ini_cnf_save_fs[-1].existing():
            cnf_path = ini_cnf_save_fs[-1].path(locs)
            ioprinter.debug_message('Removing {}'.format(cnf_path))
            shutil.rmtree(cnf_path)

    if geo_init is None:
        if 'geo_inp' in spc_dct_i:
            geo_init = spc_dct_i['geo_inp']
            ioprinter.info_message(
                'Getting initial geometry from geom dictionary')

    if geo_init is None:
        geo_init = automol.inchi.geometry(spc_dct_i['inchi'])
        ioprinter.info_message('Getting initial geometry from inchi')

    # Check if the init geometry is connected
    if geo_init is not None:
        if not automol.geom.connected(geo_init):
            geo_init = None

    return geo_init


def _functional_groups_stable(geo, thy_save_fs, mod_thy_info):
    """ look for functional group attachments that could cause
        molecule instabilities
    """

    zma = automol.geom.zmatrix(geo)
    disconn_zmas = automol.instab.product_zmas(zma)

    if disconn_zmas:
        conn_zma = automol.geom.zmatrix(geo)
        structure.instab.write_instab2(
            conn_zma, disconn_zmas,
            thy_save_fs, mod_thy_info[1:4],
            zma_locs=(0,),
            save_cnf=True)

        stable = False
    else:
        stable = True

    return stable


def _optimize_atom(spc_info, zma_init,
                   mod_thy_info, thy_run_fs,
                   cnf_save_fs, run_fs,
                   overwrite, opt_script_str, **opt_kwargs):
    """ Deal with an atom separately
    """

    geo, zma = _init_geom_opt(zma_init, spc_info, mod_thy_info,
                              run_fs, thy_run_fs,
                              opt_script_str, overwrite, **opt_kwargs)

    if geo is not None and zma is not None:
        locs = [autofile.schema.generate_new_conformer_id()]
        job = elstruct.Job.OPTIMIZATION
        filesys.save_struct(run_fs, cnf_save_fs, locs, job, mod_thy_info,
                            zma_locs=(0,), in_zma_fs=False)
        conf_found = True
    else:
        conf_found = False

    return conf_found


def _optimize_molecule(spc_info, zma_init,
                       mod_thy_info, thy_run_fs, thy_save_fs,
                       cnf_save_fs,
                       run_fs,
                       opt_script_str, overwrite,
                       kickoff_size=0.1, kickoff_backward=False,
                       **opt_kwargs):
    """ Optimize a proper geometry
    """

    # Optimize the initial geometry
    geo, zma = _init_geom_opt(
        zma_init, spc_info, mod_thy_info, run_fs, thy_run_fs,
        opt_script_str, overwrite, **opt_kwargs)

    # If connected, check for imaginary modes and fix them if possible
    if automol.geom.connected(geo):

        # Remove the imaginary mode
        geo, imag_fix_needed = _remove_imag(
            spc_info, geo, mod_thy_info, thy_run_fs,
            run_fs, kickoff_size, kickoff_backward,
            overwrite=overwrite)

        # Recheck connectivity for imag-checked geometry
        if geo is not None:

            conf_found = True
            if automol.geom.connected(geo):

                ioprinter.info_message(
                    'Saving structure as the first conformer...', newline=1)
                locs = [autofile.schema.generate_new_conformer_id()]
                job = elstruct.Job.OPTIMIZATION
                filesys.save_struct(run_fs, cnf_save_fs, locs, job, mod_thy_info,
                            zma_locs=(0,), in_zma_fs=False,
                            cart_to_zma=imag_fix_needed)

            else:

                ioprinter.info_message('Saving disconnected species...')
                _, opt_ret = es_runner.read_job(
                    job=elstruct.Job.OPTIMIZATION, run_fs=run_fs)
                structure.instab.write_instab(
                    zma_init, zma,
                    thy_save_fs, mod_thy_info[1:4],
                    opt_ret,
                    zma_locs=(0,),
                    save_cnf=True
                )

        else:

            ioprinter.warning_message('No geom found...', newline=1)
            conf_found = False

    else:

        ioprinter.info_message('Saving disconnected species...')
        conf_found = False
        _, opt_ret = es_runner.read_job(
            job=elstruct.Job.OPTIMIZATION, run_fs=run_fs)
        structure.instab.write_instab(
            zma_init, zma,
            thy_save_fs, mod_thy_info[1:4],
            opt_ret,
            zma_locs=(0,),
            save_cnf=True
        )

    # Save geom in thy filesys if a good geom is found
    if conf_found:
        thy_save_fs[-1].create(mod_thy_info[1:4])
        thy_save_path = thy_save_fs[-1].path(mod_thy_info[1:4])
        thy_save_fs[-1].file.geometry.write(geo, mod_thy_info[1:4])

        ioprinter.save_reference(thy_save_path)

    return conf_found


def _init_geom_opt(zma_init, spc_info, mod_thy_info,
                   run_fs, thy_run_fs,
                   opt_script_str, overwrite, **opt_kwargs):
    """ Generate initial geometry via optimization from either reference
    geometries or from inchi
    """

    # Set up the filesystem
    thy_run_fs[-1].create(mod_thy_info[1:4])
    thy_run_path = thy_run_fs[-1].path(mod_thy_info[1:4])

    # Call the electronic structure optimizer
    run_fs = autofile.fs.run(thy_run_path)
    es_runner.run_job(
        job=elstruct.Job.OPTIMIZATION,
        script_str=opt_script_str,
        run_fs=run_fs,
        geo=zma_init,
        spc_info=spc_info,
        thy_info=mod_thy_info,
        overwrite=overwrite,
        **opt_kwargs,
    )
    success, ret = es_runner.read_job(
        job=elstruct.Job.OPTIMIZATION, run_fs=run_fs)

    geo, zma = None, None
    if success:
        ioprinter.info_message('Succesful reference geometry optimization')
        inf_obj, _, out_str = ret
        prog = inf_obj.prog
        geo = elstruct.reader.opt_geometry(prog, out_str)
        zma = elstruct.reader.opt_zmatrix(prog, out_str)

    return geo, zma


# Functions to remove imaginary mode from a structure
def _remove_imag(spc_info, geo, mod_thy_info, thy_run_fs, run_fs,
                 kickoff_size=0.1, kickoff_backward=False,
                 overwrite=False):
    """ if there is an imaginary frequency displace geometry along the imaginary
    mode and then reoptimize
    """

    ioprinter.info_message(
        'The initial geometries will be checked for imaginary frequencies')
    script_str, opt_script_str, kwargs, opt_kwargs = qchem_params(
        *mod_thy_info[0:2])

    imag, disp_xyzs = _check_imaginary(
        spc_info, geo, mod_thy_info, thy_run_fs, script_str,
        overwrite, **kwargs)

    # Make var to fix the imaginary mode if needed to pass to other functions
    imag_fix_needed = bool(imag)

    # Make five attempts to remove imag mode if found
    chk_idx = 0
    while imag and chk_idx < 5:
        chk_idx += 1
        ioprinter.info_message(
            'Attempting kick off along mode, attempt {}...'.format(chk_idx))

        geo = _kickoff_saddle(
            geo, disp_xyzs, spc_info, mod_thy_info, run_fs, thy_run_fs,
            opt_script_str, kickoff_size, kickoff_backward,
            opt_cart=True, **opt_kwargs)

        ioprinter.info_message(
            'Removing faulty geometry from filesystem. Rerunning Hessian...')
        thy_run_path = thy_run_fs[-1].path(mod_thy_info[1:4])
        run_fs = autofile.fs.run(thy_run_path)
        run_fs[-1].remove([elstruct.Job.HESSIAN])

        ioprinter.info_message('Rerunning Hessian...')
        imag, disp_xyzs = _check_imaginary(
            spc_info, geo, mod_thy_info, thy_run_fs, script_str,
            overwrite, **kwargs)

        # Update kickoff size
        kickoff_size *= 2

    return geo, imag_fix_needed


def _check_imaginary(
        spc_info, geo, mod_thy_info, thy_run_fs, script_str,
        overwrite=False, **kwargs):
    """ check if species has an imaginary frequency
    """

    # Handle filesystem
    thy_run_fs[-1].create(mod_thy_info[1:4])
    thy_run_path = thy_run_fs[-1].path(mod_thy_info[1:4])
    run_fs = autofile.fs.run(thy_run_path)

    # Initialize info
    imag = False
    disp_xyzs = []
    hess = ((), ())

    # Run Hessian calculation
    es_runner.run_job(
        job=elstruct.Job.HESSIAN,
        spc_info=spc_info,
        thy_info=mod_thy_info,
        geo=geo,
        run_fs=run_fs,
        script_str=script_str,
        overwrite=overwrite,
        **kwargs,
        )

    # Check for imaginary modes
    success, ret = es_runner.read_job(job=elstruct.Job.HESSIAN, run_fs=run_fs)
    if success:
        inf_obj, _, out_str = ret
        prog = inf_obj.prog
        hess = elstruct.reader.hessian(prog, out_str)

        # Calculate vibrational frequencies
        if hess:
            imag = False
            _, _, imag_freq, _ = structure.vib.projrot_freqs(
                [geo], [hess], thy_run_path)
            if imag_freq:
                imag = True

            # Mode for now set the imaginary frequency check to -100:
            # Should decrease once freq projector functions properly
            if imag:
                imag = True
                ioprinter.warning_message('Imaginary mode found:')
                # norm_coos = elstruct.util.normal_coordinates(
                #     geo, hess, project=True)
                # im_norm_coo = numpy.array(norm_coos)[:, 0]
                # disp_xyzs = numpy.reshape(im_norm_coo, (-1, 3))
                norm_coos = elstruct.reader.normal_coords(prog, out_str)
                im_norm_coo = norm_coos[0]
                disp_xyzs = im_norm_coo

    return imag, disp_xyzs


def _kickoff_saddle(
        geo, disp_xyzs, spc_info, mod_thy_info, run_fs, thy_run_fs,
        opt_script_str, kickoff_size=0.1, kickoff_backward=False,
        opt_cart=True, **kwargs):
    """ kickoff from saddle to find connected minima
    """

    # Set the filesys
    thy_run_fs[-1].create(mod_thy_info[1:4])
    thy_run_path = thy_run_fs[-1].path(mod_thy_info[1:4])
    run_fs = autofile.fs.run(thy_run_path)

    # Set the displacement vectors and displace geometry
    disp_len = kickoff_size * phycon.ANG2BOHR
    if kickoff_backward:
        disp_len *= -1
    disp_xyzs = numpy.multiply(disp_xyzs, disp_len)

    ioprinter.debug_message(
        'geo test in kickoff_saddle:',
        automol.geom.string(geo), disp_xyzs)
    geo = automol.geom.displace(geo, disp_xyzs)

    # Optimize displaced geometry
    if opt_cart:
        geom = geo
    else:
        geom = automol.geom.zmatrix(geo)
    es_runner.run_job(
        job=elstruct.Job.OPTIMIZATION,
        script_str=opt_script_str,
        run_fs=run_fs,
        geo=geom,
        spc_info=spc_info,
        thy_info=mod_thy_info,
        overwrite=True,
        **kwargs,
    )

    success, ret = es_runner.read_job(
        job=elstruct.Job.OPTIMIZATION, run_fs=run_fs)
    if success:
        inf_obj, _, out_str = ret
        prog = inf_obj.prog
        geo = elstruct.reader.opt_geometry(prog, out_str)

    return geo
