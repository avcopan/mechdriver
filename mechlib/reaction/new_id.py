"""
 New reaction ID code
"""

import automol
from phydat import phycon


def build_reaction(rxn_info, thy_info, zma_locs, save_prefix):
    """
    """

    # Try to build the Z-Matrix reaction object or identify from scratch
    # zrxn = _read_from_filesys(rxn_info, thy_info, zma_locs, save_prefix)
    zrxn = None
    if zrxn is None:
        zrxn = _id_reaction(rxn_info)

    print('    Reaction class identified as: {}'.format(rxn.class))

    return rxn, zma

def _read_from_filesys(rxn_info, thy_info, zma_locs, save_prefix):
    """ Check if reaction exists in the filesystem and has been identified
    """

    mod_thy_info = thy_info  # Need to modify the theory object

    manager_prefix = [
        ('REACTION', mechanalyzer.inf.rxn.sort(rxn_info, scheme='autofile')),
        ('THEORY', mod_thy_info[1:4]),
        ('TRANSITION_STATE', ),
    ]
    cnf_save_fs = autofile.fs.manager(save_prefix, manager_prefix, 'CONFORMER')
    ini_loc_info = filesys.mincnf.min_energy_conformer_locators(
        cnf_save_fs, mod_thy_info)
    ini_min_cnf_locs, ini_min_cnf_path = ini_loc_info
    zma_fs = autofile.fs.zmatrix(ini_min_cnf_path)

    if zma_fs[-1].file.reac.exists(zma_locs):
        zrxn = zma_fs[-1].file.reac.read(zma_locs)
        zma = zma_fs[-1].file.zmatrixc.read(zma_locs)
    else:
        zma, zrxn = None, None

    return zma, zrxn


def _id_reaction(rxn_info):
    """ Identify the reaction and build the object
    """

    [rxn_ichs, _, _, _] = rxn_info   # replace with mechanalyzer grab
    rct_ichs, prd_ichs = rxn_ichs[0], rxn_ichs[1]

    rct_geos = list(map(automol.inchi.geometry, rct_ichs))
    prd_geos = list(map(automol.inchi.geometry, prd_ichs))

    rct_gras = list(map(automol.graph.without_stereo_parities,
                        map(automol.geom.graph, rct_geos)))
    prd_gras = list(map(automol.graph.without_stereo_parities,
                        map(automol.geom.graph, prd_geos)))

    rct_gras, _ = automol.graph.standard_keys_for_sequence(rct_gras)
    prd_gras, _ = automol.graph.standard_keys_for_sequence(prd_gras)

    rxns = automol.reac.find(rct_gras, prd_gras)
    rxn = rxns[0]

    rxn, rct_geos, prd_geos = (
        automol.reac.standard_keys_with_sorted_geometries(
            rxn, rct_geos, prd_geos))

    geo = automol.reac.ts_geometry(rxn, rct_geos, log=True)
    zma, zma_keys, dummy_key_dct = automol.reac.ts_zmatrix(rxn, geo)
    zrxn = automol.reac.relabel_for_zmatrix(rxn, zma_keys, dummy_key_dct)

    return zma, zrxn


# from direction
def set_reaction_direction(reacs, prods, spc_dct, cla_dct,
                           thy_info, ini_thy_info, save_prefix,
                           direction='forw'):
    """ Set the reaction of a direction
    """

    # Check if reaction is present in the class direction
    if cla_dct:
        given_class, flip_rxn = set_class_with_dct(
            cla_dct, reacs, prods)
        if flip_rxn:
            reacs, prods = prods, reacs
            rxn_info = mechanalyzer.inf.rxn.reverse(rxn_info)
    else:
        given_class = None

    # If no class, given set direction to requested direction
    if given_class is not None:
        print('    Reaction present in class dct, Setting direction to that.')
    else:
        if direction == 'forw':
            print('    User requested forward direction.')
        elif direction == 'back':
            print('    User requested reverse direction, flipping reaction.')
            reacs, prods = prods, reacs
        elif direction == 'exo':
            print('    User requested exothermic direction.',
                  'Checking energies...')
            reacs, prods = assess_rxn_ene(
                reacs, prods, spc_dct, thy_info, ini_thy_info, save_prefix)

    print('    Running reaction as:')
    print('      {} = {}'.format('+'.join(reacs), '+'.join(prods)))

    return reacs, prods, given_class


# Handle setting reaction directions with the class dictionary
def set_class_with_dct(cla_dct, reacs, prods):
    """ set the class using the class dictionary
    """
    rxn = (reacs, prods)
    rxn_rev = (prods, reacs)
    if rxn in cla_dct:
        given_class = cla_dct[rxn]
        flip_rxn = False
    elif rxn_rev in cla_dct:
        given_class = cla_dct[rxn_rev]
        flip_rxn = True
    else:
        given_class = None
        flip_rxn = False

    return given_class, flip_rxn


def parse_rxn_class_file(job_path):
    """ Read the class dictionary
    """

    if os.path.exists(os.path.join(job_path, CLA_INP)):
        print('  class.dat found. Reading contents...')
        cla_str = ptt.read_inp_str(job_path, CLA_INP, remove_comments='#')
        cla_dct = _build_cla_dct(cla_str)
    else:
        print('  No class.dat found.')
        cla_dct = {}

    return cla_dct


def _build_cla_dct(cla_str):
    """ read file
    """
    cla_dct = {}
    cla_str = remove_whitespace(cla_str)
    for line in cla_str.splitlines():
        # try:
        [rxn_line, rclass] = line.split('||')
        reacs = chemkin_io.parser.reaction.reactant_names(rxn_line)
        prods = chemkin_io.parser.reaction.product_names(rxn_line)
        cla_dct[(reacs, prods)] = rclass
        # except:
        #     print('*ERROR: Error in formatting line')
        #     print(line)
        #     sys.exit()

    return cla_dct


# Functions for the exothermicity check
def assess_rxn_ene(reacs, prods, rxn_info, thy_info, ini_thy_info, save_prefix):
    """ Check the directionality of the reaction
    """

    rxn_ene = reaction_energy(rxn_info, thy_info, ini_thy_info, save_prefix)
    method1, method2 = thy_info, ini_thy_info

    if rxn_ene is None:
        rxn_ene = reaction_energy(
            rxn_info, ini_thy_info, ini_thy_info, save_prefix)
        method1, method2 = ini_thy_info, ini_thy_info

    print('    Reaction energy is {:.2f} at {}//{} level'.format(
        rxn_ene*phycon.EH2KCAL, method1[1], method2[1]))

    if rxn_ene > 0:
        reacs, prods = prods, reacs
        rxn_info = mechanalyzer.inf.rxn.reverse(rxn_info)
        print('    Reaction is endothermic, flipping reaction.')

    return reacs, prods


def reaction_energy(rxn_info, sp_thy_info, geo_thy_info, save_prefix):
    """ reaction energy """

    rct_enes = reagent_energies(
        'reacs', rxn_info, sp_thy_info, geo_thy_info, save_prefix)
    prd_enes = reagent_energies(
        'prods', rxn_info, sp_thy_info, geo_thy_info, save_prefix)

    if rct_enes is not None and prd_enes is not None:
        rxn_ene = sum(prd_enes) - sum(rct_enes)
    else:
        rxn_ene = None

    return rxn_ene


def reagent_energies(rgt, rxn_info, sp_thy_info, geo_thy_info, save_prefix):
    """ reagent energies """

    rgt_ichs, rgt_chgs, rgt_muls = mechanalyzer.inf.rxn.rgt_info(rxn_info, rgt)

    enes = []
    for rgt_ich, rgt_chg, rgt_mul in zip(rgt_ichs, rgt_chgs, rgt_muls):

        # Set filesys
        spc_save_fs = autofile.fs.species(save_prefix)
        rgt_info = [rgt_ich, rgt_chg, rgt_mul]
        spc_save_path = spc_save_fs[-1].path(rgt_info)

        mod_geo_thy_info = modify_orb_restrict(rgt_info, geo_thy_info)
        mod_sp_thy_info = modify_orb_restrict(rgt_info, sp_thy_info)
        thy_save_fs = autofile.fs.theory(spc_save_path)
        thy_save_path = thy_save_fs[-1].path(mod_geo_thy_info[1:4])
        cnf_save_fs = autofile.fs.conformer(thy_save_path)
        min_cnf_locs, _ = min_energy_conformer_locators(
            cnf_save_fs, mod_geo_thy_info)

        # Read energy
        ene = None
        if min_cnf_locs:
            cnf_path = cnf_save_fs[-1].path(min_cnf_locs)
            sp_fs = autofile.fs.single_point(cnf_path)
            if sp_fs[-1].file.energy.exists(mod_sp_thy_info[1:4]):
                ene = sp_fs[-1].file.energy.read(mod_sp_thy_info[1:4])
        enes.append(ene)

    if any(ene is None for ene in enes):
        enes = None

    return enes
