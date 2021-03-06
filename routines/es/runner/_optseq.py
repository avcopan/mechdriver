""" Handles sequences of electronic structure jobs to deal with error handling
"""

import warnings
import automol
import elstruct
import autofile
from autoparse import pattern as app
from autoparse import find as apf
from . import _optmat as optmat


def options_matrix_optimization(script_str, prefix,
                                geom, chg, mul, method, basis, prog,
                                errors=(), options_mat=(), feedback=False,
                                frozen_coordinates=(),
                                freeze_dummy_atoms=True,
                                **kwargs):
    """ try several sets of options to generate an output file

    :returns: the input string and the output string
    :rtype: (str, str)
    """
    assert len(errors) == len(options_mat)

    subrun_fs = autofile.fs.subrun(prefix)
    max_macro_idx, _ = max(subrun_fs[-1].existing(), default=(-1, -1))
    macro_idx = max_macro_idx + 1
    micro_idx = 0
    read_geom_ = (elstruct.reader.opt_zmatrix_(prog)
                  if automol.zmatrix.is_valid(geom) else
                  elstruct.reader.opt_geometry_(prog))

    if freeze_dummy_atoms and automol.zmatrix.is_valid(geom):
        frozen_coordinates = (tuple(frozen_coordinates) +
                              automol.zmatrix.dummy_coordinate_names(geom))

    kwargs_ = dict(kwargs)
    while True:
        subrun_fs[-1].create([macro_idx, micro_idx])
        path = subrun_fs[-1].path([macro_idx, micro_idx])

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            inp_str, out_str = elstruct.run.direct(
                elstruct.writer.optimization, script_str, path,
                geom=geom, charge=chg, mult=mul, method=method,
                basis=basis, prog=prog, frozen_coordinates=frozen_coordinates,
                **kwargs_)

        error_vals = [elstruct.reader.has_error_message(prog, error, out_str)
                      for error in errors]

        # Kill the while loop if we Molpro error signaling a hopeless point
        # When an MCSCF WF calculation fails to converge at some step in opt
        # it is not clear how to save the optimization, so we give up on opt
        fail_pattern = app.one_of_these([
            app.escape('The problem occurs in Multi'),
            app.escape('The problem occurs in cipro')
        ])
        if apf.has_match(fail_pattern, out_str, case=False):
            break

        if not any(error_vals):
            # success
            break
        if not optmat.is_exhausted(options_mat):
            # try again
            micro_idx += 1
            error_row_idx = error_vals.index(True)
            kwargs_ = optmat.updated_kwargs(kwargs, options_mat)
            options_mat = optmat.advance(error_row_idx, options_mat)
            if feedback:
                geom = read_geom_(out_str)
        else:
            # failure
            warnings.resetwarnings()
            warnings.warn("elstruct robust run failed; "
                          "last run was in, {}".format(path))
            break

    return inp_str, out_str


def options_matrix_run(input_writer, script_str, prefix,
                       geom, chg, mul, method, basis, prog,
                       errors=(), options_mat=(),
                       **kwargs):
    """ try several sets of options to generate an output file

    :returns: the input string and the output string
    :rtype: (str, str)
    """
    assert len(errors) == len(options_mat)

    subrun_fs = autofile.fs.subrun(prefix)
    max_macro_idx, _ = max(subrun_fs[-1].existing(), default=(-1, -1))
    macro_idx = max_macro_idx + 1
    micro_idx = 0

    kwargs_ = dict(kwargs)
    while True:
        subrun_fs[-1].create([macro_idx, micro_idx])
        path = subrun_fs[-1].path([macro_idx, micro_idx])

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            inp_str, out_str = elstruct.run.direct(
                input_writer, script_str, path,
                geom=geom, charge=chg, mult=mul, method=method,
                basis=basis, prog=prog, **kwargs_)

        error_vals = [elstruct.reader.has_error_message(prog, error, out_str)
                      for error in errors]

        if not any(error_vals):
            # success
            break
        if not optmat.is_exhausted(options_mat):
            # try again
            micro_idx += 1
            error_row_idx = error_vals.index(True)
            kwargs_ = optmat.updated_kwargs(kwargs, options_mat)
            options_mat = optmat.advance(error_row_idx, options_mat)
        else:
            # failure
            warnings.resetwarnings()
            warnings.warn("elstruct robust run failed; "
                          "last run was in, {}".format(path))
            break

    return inp_str, out_str


def molpro_opts_mat(spc_info, geo):
    """ prepare the errors and options mat to perform successive
        single-point energy calculations in Molpro when the RHF fails to
        converge. This currently only works for doublets.
    """

    # Get the nelectrons, spins, and orbitals for the wf card
    formula = automol.geom.formula(geo)
    elec_count = automol.formula.electron_count(formula)
    two_spin = spc_info[2] - 1
    num_act_elc = two_spin
    num_act_orb = num_act_elc
    closed_orb = (elec_count - num_act_elc) // 2
    occ_orb = closed_orb + num_act_orb

    # Build the strings UHF and CASSCF wf card and set the errors and options
    uhf_str = (
        "{{uhf,maxit=300;wf,{0},1,{1};orbprint,3}}"
    ).format(elec_count, two_spin)
    cas_str = (
        "{{casscf,maxit=40;"
        "closed,{0};occ,{1};wf,{2},1,{3};canonical;orbprint,3}}"
    ).format(closed_orb, occ_orb, elec_count, two_spin)

    errors = [elstruct.Error.SCF_NOCONV]
    options_mat = [
        [{'gen_lines': {2: [uhf_str]}},
         {'gen_lines': {2: [cas_str]}},
         {'gen_lines': {2: [cas_str]}}
         ]
    ]

    return errors, options_mat
