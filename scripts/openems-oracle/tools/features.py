"""Feature-coverage sweep models (DESIGN 5.16.3).

usage: features.py WORKDIR [case...]   writes WORKDIR/cases/<name>/model.xml
Run them with sweep.py.
"""
import os
import sys

S = sys.argv[1]

LINES = ','.join(str(i) for i in range(21))


def model(props, fdtd='', bc='xmin="PEC" xmax="PEC" ymin="PEC" ymax="PEC" zmin="PEC" zmax="PEC"',
          exc='<Excitation Type="0" f0="4e9" fc="4e9"/>', grid=None, params='', bg='', cs='',
          probes=True, source=True, steps=600, ts='TimeStep="1.5e-12"', args=None, comment=''):
    if grid is None:
        grid = f'<RectilinearGrid DeltaUnit="1e-3" CoordSystem="0"><XLines>{LINES}</XLines><YLines>{LINES}</YLines><ZLines>{LINES}</ZLines></RectilinearGrid>'
    src = ''
    if source:
        src = '''<Excitation Name="exc" Type="0" Excite="0,0,1">
        <Primitives><Box Priority="0"><P1 X="4" Y="10" Z="8"/><P2 X="4" Y="10" Z="12"/></Box></Primitives>
      </Excitation>'''
    pr = ''
    if probes:
        pr = '''<ProbeBox Name="e1" Type="2"><Primitives><Box Priority="0"><P1 X="16" Y="10" Z="10"/><P2 X="16" Y="10" Z="10"/></Box></Primitives></ProbeBox>
      <ProbeBox Name="e2" Type="2"><Primitives><Box Priority="0"><P1 X="10" Y="5" Z="7"/><P2 X="10" Y="5" Z="7"/></Box></Primitives></ProbeBox>
      <ProbeBox Name="e3" Type="2"><Primitives><Box Priority="0"><P1 X="11" Y="13" Z="12"/><P2 X="11" Y="13" Z="12"/></Box></Primitives></ProbeBox>
      <ProbeBox Name="h1" Type="3"><Primitives><Box Priority="0"><P1 X="12" Y="12" Z="10"/><P2 X="12" Y="12" Z="10"/></Box></Primitives></ProbeBox>
      <ProbeBox Name="u1" Type="0"><Primitives><Box Priority="0"><P1 X="16" Y="6" Z="8"/><P2 X="16" Y="6" Z="12"/></Box></Primitives></ProbeBox>'''
    head = '<?xml version="1.0" encoding="UTF-8"?>\n'
    if args:
        head += f'<!-- args: {args} -->\n'
    if comment:
        head += f'<!-- {comment} -->\n'
    return f'''{head}<openEMS>
  <FDTD NumberOfTimesteps="{steps}" endCriteria="1e-30" {ts} {fdtd}>
    {exc}
    <BoundaryCond {bc}/>
  </FDTD>
  <ContinuousStructure CoordSystem="0"{cs}>
    {params}
    {grid}
    {bg}
    <Properties>
      {src}
      {pr}
      {props}
    </Properties>
  </ContinuousStructure>
</openEMS>
'''


def mat(prims, eps='4', extra='', name='m', weight=''):
    w = f'<Weight {weight}/>' if weight else ''
    return f'<Material Name="{name}" {extra}><Property Epsilon="{eps}"/>{w}<Primitives>{prims}</Primitives></Material>'


def metal(prims, name='pec'):
    return f'<Metal Name="{name}"><Primitives>{prims}</Primitives></Metal>'


def box(p1, p2, pri=0, extra='', inner=''):
    return (f'<Box Priority="{pri}" {extra}><P1 X="{p1[0]}" Y="{p1[1]}" Z="{p1[2]}"/>'
            f'<P2 X="{p2[0]}" Y="{p2[1]}" Z="{p2[2]}"/>{inner}</Box>')


def pt(tag, p):
    return f'<{tag} X="{p[0]}" Y="{p[1]}" Z="{p[2]}"/>'


def tr(*ops):
    return '<Transformation>' + ''.join(f'<{n} Argument="{a}"/>' for n, a in ops) + '</Transformation>'


C = {}

# ------------------------------------------------------------ primitives
C['prim_box_mat'] = model(mat(box((7, 7, 7), (13, 13, 13))))
C['prim_box_offgrid'] = model(mat(box((7.3, 6.6, 7.5), (12.4, 13.2, 12.7))))
C['prim_box_metal'] = model(metal(box((8, 8, 6), (12, 12, 14))))
C['prim_box_metal_sheet'] = model(metal(box((8, 6, 10), (13, 14, 10))))
C['prim_box_metal_line'] = model(metal(box((8, 10, 6), (8, 10, 14))))
C['prim_box_reversed'] = model(mat(box((13, 13, 13), (7, 7, 7))))
C['prim_multibox'] = model(f'<Material Name="m"><Property Epsilon="4"/><Primitives><MultiBox Priority="0" QtyBox="2">'
                           f'{pt("StartP", (6, 6, 6))}{pt("EndP", (9, 14, 14))}{pt("StartP", (11, 6, 6))}{pt("EndP", (14, 14, 9))}'
                           '</MultiBox></Primitives></Material>')
C['prim_sphere_mat'] = model(mat(f'<Sphere Priority="0" Radius="4.5">{pt("Center", (10, 10, 10))}</Sphere>'))
C['prim_sphere_metal'] = model(metal(f'<Sphere Priority="0" Radius="3.5">{pt("Center", (10.2, 9.7, 10.4))}</Sphere>'))
C['prim_sphshell'] = model(mat(f'<SphericalShell Priority="0" Radius="4" ShellWidth="2">{pt("Center", (10, 10, 10))}</SphericalShell>', eps='6'))
C['prim_sphshell_metal'] = model(metal(f'<SphericalShell Priority="0" Radius="5" ShellWidth="1">{pt("Center", (10, 10, 10))}</SphericalShell>'))
C['prim_cyl_z'] = model(mat(f'<Cylinder Priority="0" Radius="3.5">{pt("P1", (10, 10, 5))}{pt("P2", (10, 10, 15))}</Cylinder>'))
C['prim_cyl_oblique'] = model(mat(f'<Cylinder Priority="0" Radius="2.5">{pt("P1", (6, 6, 6))}{pt("P2", (14, 13, 14))}</Cylinder>'))
C['prim_cyl_metal'] = model(metal(f'<Cylinder Priority="0" Radius="2">{pt("P1", (10, 5, 10))}{pt("P2", (10, 15, 10))}</Cylinder>'))
C['prim_cyl_thin_metal'] = model(metal(f'<Cylinder Priority="0" Radius="0.2">{pt("P1", (10, 10, 4))}{pt("P2", (10, 10, 16))}</Cylinder>'))
C['prim_cylshell'] = model(mat(f'<CylindricalShell Priority="0" Radius="4" ShellWidth="2">{pt("P1", (10, 10, 4))}{pt("P2", (10, 10, 16))}</CylindricalShell>', eps='5'))
C['prim_cylshell_metal'] = model(metal(f'<CylindricalShell Priority="0" Radius="5" ShellWidth="1">{pt("P1", (10, 10, 4))}{pt("P2", (10, 10, 16))}</CylindricalShell>'))
POLY = '<Vertex X1="6" X2="6"/><Vertex X1="14" X2="7"/><Vertex X1="12" X2="14"/><Vertex X1="7" X2="12"/>'
C['prim_polygon_metal'] = model(metal(f'<Polygon Priority="0" NormDir="2" Elevation="10" QtyVertices="4">{POLY}</Polygon>'))
C['prim_polygon_x'] = model(metal(f'<Polygon Priority="0" NormDir="0" Elevation="9" QtyVertices="4">{POLY}</Polygon>'))
C['prim_linpoly'] = model(mat(f'<LinPoly Priority="0" NormDir="2" Elevation="7" Length="6" QtyVertices="4">{POLY}</LinPoly>'))
C['prim_linpoly_y'] = model(mat(f'<LinPoly Priority="0" NormDir="1" Elevation="7" Length="5" QtyVertices="4">{POLY}</LinPoly>'))
C['prim_linpoly_metal'] = model(metal(f'<LinPoly Priority="0" NormDir="2" Elevation="8" Length="3" QtyVertices="4">{POLY}</LinPoly>'))
RP = '<Vertex X1="2" X2="6"/><Vertex X1="5" X2="6"/><Vertex X1="5" X2="14"/><Vertex X1="2" X2="14"/>'
# rotational polygon: NormDir=0 means vertices in (y, z)? axis rot z
C['prim_rotpoly'] = model(mat(f'<RotPoly Priority="0" NormDir="0" RotAxisDir="2" QtyVertices="4">{RP}<Angles Start="0" Stop="6.283185307179586"/>'
                              f'{tr(("Translate", "10,10,0"))}</RotPoly>'))
C['prim_rotpoly_partial'] = model(mat(f'<RotPoly Priority="0" NormDir="0" RotAxisDir="2" QtyVertices="4">{RP}<Angles Start="0" Stop="3.14159265"/>'
                                      f'{tr(("Translate", "10,10,0"))}</RotPoly>'))
CUBE_V = ''.join(f'<Vertex>{x},{y},{z}</Vertex>' for x, y, z in
                 [(7, 7, 7), (13, 7, 7), (13, 13, 7), (7, 13, 7), (7, 7, 13), (13, 7, 13), (13, 13, 13), (7, 13, 13)])
CUBE_F = ''.join(f'<Face>{f}</Face>' for f in
                 ['0,2,1', '0,3,2', '4,5,6', '4,6,7', '0,1,5', '0,5,4', '1,2,6', '1,6,5', '2,3,7', '2,7,6', '3,0,4', '3,4,7'])
C['prim_polyhedron_cube'] = model(mat(f'<Polyhedron Priority="0">{CUBE_V}{CUBE_F}</Polyhedron>'))
TET_V = ''.join(f'<Vertex>{x},{y},{z}</Vertex>' for x, y, z in [(6, 6, 6), (15, 7, 7), (8, 15, 8), (9, 9, 15)])
TET_F = ''.join(f'<Face>{f}</Face>' for f in ['0,2,1', '0,1,3', '1,2,3', '0,3,2'])
C['prim_polyhedron_tet'] = model(mat(f'<Polyhedron Priority="0">{TET_V}{TET_F}</Polyhedron>'))
C['prim_polyhedron_metal'] = model(metal(f'<Polyhedron Priority="0">{TET_V}{TET_F}</Polyhedron>'))
C['prim_stl'] = model(mat('<PolyhedronReader Priority="0" FileName="tet.stl" FileType="STL"/>'))
C['prim_ply'] = model(mat('<PolyhedronReader Priority="0" FileName="tet.ply" FileType="PLY"/>'))
C['prim_curve_metal'] = model(metal(f'<Curve Priority="0">{pt("Vertex", (6, 10, 6))}{pt("Vertex", (6, 10, 14))}{pt("Vertex", (12, 10, 14))}</Curve>'))
C['prim_curve_diag'] = model(metal(f'<Curve Priority="0">{pt("Vertex", (6, 6, 6))}{pt("Vertex", (13, 12, 14))}</Curve>'))
C['prim_wire'] = model(metal(f'<Wire Priority="0" WireRadius="0.8">{pt("Vertex", (10, 10, 5))}{pt("Vertex", (10, 10, 15))}</Wire>'))
C['prim_wire_thick'] = model(metal(f'<Wire Priority="0" WireRadius="2">{pt("Vertex", (10, 6, 10))}{pt("Vertex", (10, 14, 10))}</Wire>'))
C['prim_userdefined'] = model(mat('<UserDefined Priority="0"><Equation>(x-10)*(x-10)+(y-10)*(y-10)+(z-10)*(z-10)&lt;16</Equation></UserDefined>'))
C['prim_point_probe'] = model('<ProbeBox Name="pp" Type="2"><Primitives><Point Priority="0" X="13" Y="9" Z="11"/></Primitives></ProbeBox>')
C['prim_cs_cyl_box'] = model(mat(f'<Box Priority="0" CoordSystem="1">{pt("P1", (2, 0, 6))}{pt("P2", (5, 1.5, 14))}'
                                 f'{tr(("Translate", "10,10,0"))}</Box>'))
# ------------------------------------------------------------ priorities
C['prio_overlap'] = model(mat(box((6, 6, 6), (14, 14, 14), 1), name='a') +
                          mat(box((9, 9, 4), (11, 11, 16), 5), eps='9', name='b'))
C['prio_overlap_rev'] = model(mat(box((6, 6, 6), (14, 14, 14), 5), name='a') +
                              mat(box((9, 9, 4), (11, 11, 16), 1), eps='9', name='b'))
C['prio_metal_in_mat'] = model(mat(box((6, 6, 6), (14, 14, 14), 1)) + metal(box((9, 9, 9), (11, 11, 15), 10)))
C['prio_mat_over_metal'] = model(mat(box((6, 6, 6), (14, 14, 14), 10)) + metal(box((9, 9, 4), (11, 11, 15), 1)))
C['prio_equal'] = model(mat(box((6, 6, 6), (12, 12, 12), 2), name='a') + mat(box((9, 9, 9), (14, 14, 14), 2), eps='9', name='b'))
# ------------------------------------------------------------ transforms
SM = box((-2, -2, -3), (2, 2, 3))
C['tr_translate'] = model(mat(box((-2, -2, -3), (2, 2, 3), inner=tr(('Translate', '10,10,10')))))
C['tr_rotate_z'] = model(mat(box((-2, -4, -3), (2, 4, 3), inner=tr(('Rotate_Z', '0.5'), ('Translate', '10,10,10')))))
C['tr_rotate_x'] = model(mat(box((-2, -4, -3), (2, 4, 3), inner=tr(('Rotate_X', '0.7'), ('Translate', '10,10,10')))))
C['tr_rotate_y'] = model(mat(box((-2, -4, -3), (2, 4, 3), inner=tr(('Rotate_Y', '-0.4'), ('Translate', '10,10,10')))))
C['tr_rotate_origin'] = model(mat(box((-2, -4, -3), (2, 4, 3), inner=tr(('Rotate_Origin', '1,1,0,0.6'), ('Translate', '10,10,10')))))
C['tr_scale'] = model(mat(box((-1, -1, -1.5), (1, 1, 1.5), inner=tr(('Scale', '2'), ('Translate', '10,10,10')))))
C['tr_scale3'] = model(mat(box((-1, -1, -1), (1, 1, 1), inner=tr(('Scale3', '2,3,4'), ('Translate', '10,10,10')))))
C['tr_matrix'] = model(mat(box((-2, -4, -3), (2, 4, 3), inner=tr(('Matrix', '0.8,-0.6,0,10,0.6,0.8,0,10,0,0,1,10,0,0,0,1')))))
C['tr_order'] = model(mat(box((0, 0, 0), (4, 2, 6), inner=tr(('Translate', '8,9,7'), ('Rotate_Z', '0.3')))))
C['tr_sphere'] = model(mat(f'<Sphere Priority="0" Radius="2">{pt("Center", (0, 0, 0))}{tr(("Scale3", "2,1.5,1"), ("Translate", "10,10,10"))}</Sphere>'))
C['tr_metal'] = model(metal(box((-3, -3, 0), (3, 3, 0), inner=tr(('Rotate_X', '0.4'), ('Translate', '10,10,10')))))
C['tr_cylinder'] = model(mat(f'<Cylinder Priority="0" Radius="2.5">{pt("P1", (0, 0, -5))}{pt("P2", (0, 0, 5))}{tr(("Rotate_Y", "0.785398"), ("Translate", "10,10,10"))}</Cylinder>'))
# ------------------------------------------------------------ materials
C['mat_kappa'] = model(mat(box((7, 7, 7), (13, 13, 13)), extra='') .replace('Epsilon="4"', 'Epsilon="2" Kappa="0.05"'))
C['mat_mue_sigma'] = model(mat(box((7, 7, 7), (13, 13, 13))).replace('Epsilon="4"', 'Mue="3" Sigma="2000"'))
C['mat_aniso'] = model(mat(box((7, 7, 7), (13, 13, 13)), extra='Isotropy="0"').replace('Epsilon="4"', 'Epsilon="2,4,6" Mue="1,2,1.5" Kappa="0,0.01,0.02"'))
C['mat_aniso_iso1'] = model(mat(box((7, 7, 7), (13, 13, 13)), extra='Isotropy="1"').replace('Epsilon="4"', 'Epsilon="2,4,6"'))
C['mat_aniso_noattr'] = model(mat(box((7, 7, 7), (13, 13, 13))).replace('Epsilon="4"', 'Epsilon="2,4,6"'))
C['mat_weight'] = model(mat(box((5, 5, 5), (15, 15, 15)), eps='4', weight='Epsilon="1+0.1*x"'))
C['mat_weight_aniso'] = model(mat(box((5, 5, 5), (15, 15, 15)), eps='2,3,4', extra='Isotropy="0"',
                                  weight='Epsilon="1+0.1*x,1+0.05*y,1" Kappa="z*0.001"').replace('Epsilon="2,3,4"', 'Epsilon="2,3,4" Kappa="1"'))
C['mat_weight_coords'] = model(mat(box((5, 5, 5), (15, 15, 15)), eps='4', weight='Epsilon="1+sin(x*0.3)*cos(y*0.2)+z/20"'))
C['mat_weight_rho'] = model(mat(box((5, 5, 5), (15, 15, 15)), eps='4', weight='Epsilon="1+0.05*rho"'))
C['mat_density'] = model(mat(box((7, 7, 7), (13, 13, 13))).replace('Epsilon="4"', 'Epsilon="4" Kappa="0.1" Density="1000"'))
C['mat_lowtail'] = model(mat(box((7, 7, 7), (13, 13, 13))).replace('Epsilon="4"', 'Epsilon="0.5"'))
C['bg_material'] = model('', bg='<BackgroundMaterial Epsilon="2.5" Mue="1.2" Kappa="0.001"/>')
C['bg_material_with_box'] = model(mat(box((7, 7, 7), (13, 13, 13))), bg='<BackgroundMaterial Epsilon="2"/>')
# ------------------------------------------------------------ parameters / expressions
C['param_set'] = model(mat('<Box Priority="0"><P1 X="cx-w" Y="cy-w" Z="7"/><P2 X="cx+w" Y="cy+w" Z="7+2*w"/></Box>').replace('Epsilon="4"', 'Epsilon="er"'),
                       params='<ParameterSet><Parameter Type="Const" name="cx" value="10"/><Parameter Type="Const" name="cy" value="10.5"/>'
                              '<Parameter Type="Const" name="w" value="3"/><Parameter Type="Const" name="er" value="3.3"/></ParameterSet>')
C['param_term'] = model(mat('<Box Priority="0"><P1 X="term:5+2" Y="7" Z="7"/><P2 X="13" Y="term:2*6.5" Z="13"/></Box>'))
C['param_grid_expr'] = model(mat(box((7, 7, 7), (13, 13, 13))),
                             grid='<RectilinearGrid DeltaUnit="1e-3" CoordSystem="0"><XLines>' + ','.join(f'{i}*1' for i in range(21)) +
                             f'</XLines><YLines>{LINES}</YLines><ZLines>{LINES}</ZLines></RectilinearGrid>')
C['grid_unsorted'] = model(mat(box((7, 7, 7), (13, 13, 13))),
                           grid='<RectilinearGrid DeltaUnit="1e-3" CoordSystem="0"><XLines>' + ','.join(str(i) for i in reversed(range(21))) +
                           f'</XLines><YLines>{LINES},5,10</YLines><ZLines>{LINES}</ZLines></RectilinearGrid>')
C['grid_deltaunit_m'] = model('''<Material Name="m"><Property Epsilon="4"/><Primitives><Box Priority="0"><P1 X="0.007" Y="0.007" Z="0.007"/><P2 X="0.013" Y="0.013" Z="0.013"/></Box></Primitives></Material>''',
                              grid='<RectilinearGrid DeltaUnit="1" CoordSystem="0"><XLines>' + ','.join(f'{i*1e-3:g}' for i in range(21)) +
                              '</XLines><YLines>' + ','.join(f'{i*1e-3:g}' for i in range(21)) + '</YLines><ZLines>' +
                              ','.join(f'{i*1e-3:g}' for i in range(21)) + '</ZLines></RectilinearGrid>', probes=False, source=False).replace(
    '<Properties>', '''<Properties><Excitation Name="exc" Type="0" Excite="0,0,1"><Primitives><Box Priority="0"><P1 X="0.004" Y="0.010" Z="0.008"/><P2 X="0.004" Y="0.010" Z="0.012"/></Box></Primitives></Excitation>
      <ProbeBox Name="e1" Type="2"><Primitives><Box Priority="0"><P1 X="0.016" Y="0.010" Z="0.010"/><P2 X="0.016" Y="0.010" Z="0.010"/></Box></Primitives></ProbeBox>
      <ProbeBox Name="u1" Type="0"><Primitives><Box Priority="0"><P1 X="0.016" Y="0.006" Z="0.008"/><P2 X="0.016" Y="0.006" Z="0.012"/></Box></Primitives></ProbeBox>''')
GRADED = ','.join(f'{v:g}' for v in [0, 1.5, 3, 4, 5, 6, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13, 14, 15.5, 17, 18.5, 20])
C['grid_graded'] = model(mat(box((7.5, 8, 7), (12, 13, 13))),
                         grid=f'<RectilinearGrid DeltaUnit="1e-3" CoordSystem="0"><XLines>{GRADED}</XLines><YLines>{GRADED}</YLines><ZLines>{LINES}</ZLines></RectilinearGrid>',
                         ts='TimeStep="9e-13"')
# ------------------------------------------------------------ excitations
C['exc_weight'] = model('', source=False).replace('<Properties>', '''<Properties><Excitation Name="exc" Type="0" Excite="0,0,1">
        <Weight X="0" Y="0" Z="sin(3.14159*y/20)*(1+x/10)"/>
        <Primitives><Box Priority="0"><P1 X="4" Y="0" Z="8"/><P2 X="6" Y="20" Z="12"/></Box></Primitives></Excitation>''')
C['exc_vector'] = model('', source=False).replace('<Properties>', '''<Properties><Excitation Name="exc" Type="0" Excite="0.3,-0.5,1">
        <Primitives><Box Priority="0"><P1 X="4" Y="8" Z="8"/><P2 X="6" Y="11" Z="12"/></Box></Primitives></Excitation>''')
C['exc_delay'] = model('', source=False).replace('<Properties>', '''<Properties><Excitation Name="exc" Type="0" Excite="0,0,1" Delay="1e-10">
        <Primitives><Box Priority="0"><P1 X="4" Y="10" Z="8"/><P2 X="4" Y="10" Z="12"/></Box></Primitives></Excitation>''')
C['exc_disabled'] = model('', source=True).replace('<Properties>', '''<Properties><Excitation Name="exc2" Type="0" Excite="0,1,0" Enabled="0">
        <Primitives><Box Priority="0"><P1 X="14" Y="6" Z="10"/><P2 X="14" Y="12" Z="10"/></Box></Primitives></Excitation>''')
C['exc_hard_e'] = model('', source=False).replace('<Properties>', '''<Properties><Excitation Name="exc" Type="1" Excite="0,0,1">
        <Primitives><Box Priority="0"><P1 X="4" Y="10" Z="8"/><P2 X="4" Y="10" Z="12"/></Box></Primitives></Excitation>''')
C['exc_soft_h'] = model('', source=False).replace('<Properties>', '''<Properties><Excitation Name="exc" Type="2" Excite="0,0,1">
        <Primitives><Box Priority="0"><P1 X="4" Y="10" Z="8"/><P2 X="4" Y="10" Z="12"/></Box></Primitives></Excitation>''')
C['exc_two'] = model('', source=True).replace('<Properties>', '''<Properties><Excitation Name="exc2" Type="0" Excite="1,0,0">
        <Primitives><Box Priority="0"><P1 X="12" Y="14" Z="6"/><P2 X="15" Y="14" Z="6"/></Box></Primitives></Excitation>''')
C['exc_on_metal'] = model(metal(box((4, 10, 9), (4, 10, 11), 5)))
C['exc_box_3d'] = model('', source=False).replace('<Properties>', '''<Properties><Excitation Name="exc" Type="0" Excite="1,1,1">
        <Primitives><Box Priority="0"><P1 X="4" Y="8" Z="8"/><P2 X="6" Y="11" Z="11"/></Box></Primitives></Excitation>''')
C['exc_offgrid'] = model('', source=False).replace('<Properties>', '''<Properties><Excitation Name="exc" Type="0" Excite="0,0,1">
        <Primitives><Box Priority="0"><P1 X="4.3" Y="9.6" Z="8.2"/><P2 X="4.3" Y="9.6" Z="11.7"/></Box></Primitives></Excitation>''')
C['exc_sphere'] = model('', source=False).replace('<Properties>', '''<Properties><Excitation Name="exc" Type="0" Excite="0,0,1">
        <Primitives><Sphere Priority="0" Radius="2"><Center X="6" Y="10" Z="10"/></Sphere></Primitives></Excitation>''')
C['exc_curve'] = model('', source=False).replace('<Properties>', '''<Properties><Excitation Name="exc" Type="0" Excite="0,0,1">
        <Primitives><Curve Priority="0"><Vertex X="5" Y="10" Z="8"/><Vertex X="5" Y="10" Z="12"/></Curve></Primitives></Excitation>''')
# ------------------------------------------------------------ probes
C['probe_weight'] = model('<ProbeBox Name="uw" Type="0" Weight="-2.5"><Primitives><Box Priority="0"><P1 X="16" Y="6" Z="8"/><P2 X="16" Y="6" Z="12"/></Box></Primitives></ProbeBox>'
                          '<ProbeBox Name="ew" Type="2" Weight="3"><Primitives><Box Priority="0"><P1 X="16" Y="10" Z="10"/><P2 X="16" Y="10" Z="10"/></Box></Primitives></ProbeBox>')
C['probe_e_box'] = model('<ProbeBox Name="eb" Type="2"><Primitives><Box Priority="0"><P1 X="12" Y="9" Z="9"/><P2 X="14" Y="11" Z="11"/></Box></Primitives></ProbeBox>'
                         '<ProbeBox Name="hb" Type="3"><Primitives><Box Priority="0"><P1 X="12" Y="9" Z="9"/><P2 X="14" Y="11" Z="11"/></Box></Primitives></ProbeBox>')
C['probe_e_offgrid'] = model('<ProbeBox Name="eo" Type="2"><Primitives><Box Priority="0"><P1 X="12.4" Y="9.6" Z="9.3"/><P2 X="12.4" Y="9.6" Z="9.3"/></Box></Primitives></ProbeBox>'
                             '<ProbeBox Name="ho" Type="3"><Primitives><Box Priority="0"><P1 X="12.7" Y="9.2" Z="9.5"/><P2 X="12.7" Y="9.2" Z="9.5"/></Box></Primitives></ProbeBox>')
C['probe_u_diag'] = model('<ProbeBox Name="ud" Type="0"><Primitives><Box Priority="0"><P1 X="12" Y="8" Z="8"/><P2 X="15" Y="11" Z="12"/></Box></Primitives></ProbeBox>')
C['probe_u_rev'] = model('<ProbeBox Name="ur" Type="0"><Primitives><Box Priority="0"><P1 X="16" Y="6" Z="12"/><P2 X="16" Y="6" Z="8"/></Box></Primitives></ProbeBox>')
C['probe_i_dirs'] = model(metal(box((10, 10, 3), (10, 10, 17))) + ''.join(
    f'<ProbeBox Name="i{n}" Type="1" NormDir="{nd}"><Primitives><Box Priority="0"><P1 X="{a[0]}" Y="{a[1]}" Z="{a[2]}"/><P2 X="{b[0]}" Y="{b[1]}" Z="{b[2]}"/></Box></Primitives></ProbeBox>'
    for n, nd, a, b in [('z', 2, (8, 8, 12), (12, 12, 12)), ('zr', 2, (12, 12, 12), (8, 8, 12)), ('x', 0, (12, 6, 6), (12, 14, 14)), ('nz', 2, (8.5, 8.5, 12.5), (11.5, 11.5, 12.5))]))
C['probe_fd'] = model('<ProbeBox Name="uf" Type="0"><FD_Samples>2e9,4e9,6e9</FD_Samples><Primitives><Box Priority="0"><P1 X="16" Y="6" Z="8"/><P2 X="16" Y="6" Z="12"/></Box></Primitives></ProbeBox>'
                      '<ProbeBox Name="ef" Type="2"><FD_Samples>3e9,5e9</FD_Samples><Primitives><Box Priority="0"><P1 X="16" Y="10" Z="10"/><P2 X="16" Y="10" Z="10"/></Box></Primitives></ProbeBox>'
                      '<ProbeBox Name="hf" Type="3"><FD_Samples>3e9,5e9</FD_Samples><Primitives><Box Priority="0"><P1 X="12" Y="12" Z="10"/><P2 X="12" Y="12" Z="10"/></Box></Primitives></ProbeBox>'
                      + metal(box((10, 10, 3), (10, 10, 17))) +
                      '<ProbeBox Name="if" Type="1" NormDir="2"><FD_Samples>3e9,5e9</FD_Samples><Primitives><Box Priority="0"><P1 X="8" Y="8" Z="12"/><P2 X="12" Y="12" Z="12"/></Box></Primitives></ProbeBox>')
C['probe_times'] = model('<ProbeBox Name="ut" Type="0" StartTime="2e-10" StopTime="6e-10"><Primitives><Box Priority="0"><P1 X="16" Y="6" Z="8"/><P2 X="16" Y="6" Z="12"/></Box></Primitives></ProbeBox>'
                         '<ProbeBox Name="et" Type="2" StartTime="1e-10"><Primitives><Box Priority="0"><P1 X="16" Y="10" Z="10"/><P2 X="16" Y="10" Z="10"/></Box></Primitives></ProbeBox>')
C['probe_cylinder_prim'] = model('<ProbeBox Name="uc" Type="0"><Primitives><Cylinder Priority="0" Radius="0"><P1 X="16" Y="6" Z="8"/><P2 X="16" Y="6" Z="12"/></Cylinder></Primitives></ProbeBox>')
C['probe_two_prims'] = model('<ProbeBox Name="u2" Type="0"><Primitives><Box Priority="0"><P1 X="16" Y="6" Z="8"/><P2 X="16" Y="6" Z="12"/></Box><Box Priority="0"><P1 X="14" Y="6" Z="8"/><P2 X="14" Y="6" Z="12"/></Box></Primitives></ProbeBox>')
C['probe_transformed'] = model('<ProbeBox Name="ut" Type="0"><Primitives><Box Priority="0"><P1 X="0" Y="0" Z="-2"/><P2 X="0" Y="0" Z="2"/>' + tr(('Translate', '16,6,10')) + '</Box></Primitives></ProbeBox>')
# ------------------------------------------------------------ FDTD settings
C['fdtd_oversampling'] = model('', fdtd='OverSampling="10" f_max="8e9"')
C['fdtd_oversampling1'] = model('', fdtd='OverSampling="1"')
C['fdtd_timestepfactor'] = model('', ts='TimeStepFactor="0.5"')
C['fdtd_tsm1'] = model(mat(box((7, 7, 7), (13, 13, 13))), ts='TimeStepMethod="1"')
C['fdtd_tsm3_graded'] = model(mat(box((7, 7, 7), (13, 13, 13))), ts='',
                              grid=f'<RectilinearGrid DeltaUnit="1e-3" CoordSystem="0"><XLines>{GRADED}</XLines><YLines>{LINES}</YLines><ZLines>{GRADED}</ZLines></RectilinearGrid>')
C['fdtd_maxtime'] = model('', fdtd='MaxTime="5e-10"', steps=100000)
C['fdtd_cellconst'] = model(mat(box((7.3, 6.6, 7.5), (12.4, 13.2, 12.7))), fdtd='CellConstantMaterial="1"')
C['fdtd_ts_and_steps0'] = model('', steps=0)
C['fdtd_endcrit'] = model('', fdtd='', bc='xmin="MUR" xmax="MUR" ymin="MUR" ymax="MUR" zmin="MUR" zmax="MUR"', steps=5000).replace('endCriteria="1e-30"', 'endCriteria="1e-3"')
C['fdtd_coordsystem_attr'] = model('', fdtd='CoordSystem="0"')
# ------------------------------------------------------------ boundaries
for n, bc in [('bc_numbers', 'xmin="0" xmax="1" ymin="2" ymax="0" zmin="1" zmax="2"'),
              ('bc_names_lower', 'xmin="pec" xmax="pmc" ymin="mur" ymax="PEC" zmin="PMC" zmax="MUR"'),
              ('bc_pml_n', 'xmin="PML_4" xmax="PML_6" ymin="PEC" ymax="PEC" zmin="PEC" zmax="PEC"'),
              ('bc_pml_3', 'xmin="3" xmax="PEC" ymin="PEC" ymax="PEC" zmin="PEC" zmax="PEC"'),
              ('bc_mur_pv', 'xmin="MUR" xmax="MUR" ymin="MUR" ymax="MUR" zmin="MUR" zmax="MUR" MUR_PhaseVelocity="2e8"'),
              ('bc_mixed', 'xmin="PMC" xmax="MUR" ymin="PEC" ymax="PMC" zmin="MUR" zmax="PEC"')]:
    C[n] = model('', bc=bc)
C['bc_pml_mat'] = model(mat(box((-1, 7, 7), (21, 13, 13))), bc='xmin="PML_6" xmax="PML_6" ymin="PEC" ymax="PEC" zmin="PEC" zmax="PEC"')
# ------------------------------------------------------------ misc properties
C['lumped_caps'] = model('<LumpedElement Name="R" Direction="2" Caps="0" R="50"><Primitives><Box Priority="0"><P1 X="14" Y="10" Z="9"/><P2 X="14" Y="11" Z="11"/></Box></Primitives></LumpedElement>')
C['lumped_c_only'] = model('<LumpedElement Name="C" Direction="2" C="1e-12"><Primitives><Box Priority="0"><P1 X="14" Y="10" Z="9"/><P2 X="14" Y="10" Z="11"/></Box></Primitives></LumpedElement>')
C['abc_sheet'] = model('<AbsorbingBC Name="abc" NormalSignPositive="1" PhaseVelocity="299792458" AbsorbingBoundaryType="0"><Primitives><Box Priority="0"><P1 X="18" Y="0" Z="0"/><P2 X="18" Y="20" Z="20"/></Box></Primitives></AbsorbingBC>')
C['cond_sheet'] = model('<ConductingSheet Name="cs" Conductivity="5.8e7" Thickness="35e-6"><Primitives><Box Priority="0"><P1 X="8" Y="6" Z="10"/><P2 X="13" Y="14" Z="10"/></Box></Primitives></ConductingSheet>')
C['unknown_prop'] = model('<Foo Name="foo"><Primitives><Box Priority="0"><P1 X="8" Y="6" Z="10"/><P2 X="13" Y="14" Z="10"/></Box></Primitives></Foo>')
C['prop_no_prims'] = model('<Material Name="empty"><Property Epsilon="4"/></Material><Metal Name="m2"/>')
C['dump_e_vtk'] = model('<DumpBox Name="Et" DumpType="0" DumpMode="0" FileType="1"><Primitives><Box Priority="0"><P1 X="8" Y="8" Z="10"/><P2 X="12" Y="12" Z="10"/></Box></Primitives></DumpBox>', steps=60)
C['dump_subsample'] = model('<DumpBox Name="Es" DumpType="0" DumpMode="1" FileType="1" SubSampling="2,3,1"><Primitives><Box Priority="0"><P1 X="2" Y="2" Z="10"/><P2 X="18" Y="18" Z="10"/></Box></Primitives></DumpBox>', steps=60)
C['dump_optres'] = model('<DumpBox Name="Eo" DumpType="0" DumpMode="1" FileType="1" OptResolution="3"><Primitives><Box Priority="0"><P1 X="2" Y="2" Z="10"/><P2 X="18" Y="18" Z="10"/></Box></Primitives></DumpBox>', steps=60)
C['dump_fd_h'] = model('<DumpBox Name="Hf" DumpType="11" DumpMode="2" FileType="1"><FD_Samples>3e9,5e9</FD_Samples><Primitives><Box Priority="0"><P1 X="2" Y="2" Z="10"/><P2 X="18" Y="18" Z="11"/></Box></Primitives></DumpBox>')
C['dump_nomode'] = model('<DumpBox Name="Hn" DumpType="1" FileType="1"><Primitives><Box Priority="0"><P1 X="6" Y="6" Z="10"/><P2 X="12" Y="12" Z="10"/></Box></Primitives></DumpBox>', steps=40)

# STL / PLY files for the readers (same tetrahedron as prim_polyhedron_tet)
TV = [(6, 6, 6), (15, 7, 7), (8, 15, 8), (9, 9, 15)]
TF = [(0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)]


def stl():
    s = 'solid tet\n'
    for f in TF:
        s += ' facet normal 0 0 0\n  outer loop\n' + ''.join(f'   vertex {TV[i][0]} {TV[i][1]} {TV[i][2]}\n' for i in f) + '  endloop\n endfacet\n'
    return s + 'endsolid tet\n'


def ply():
    s = 'ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nelement face 4\nproperty list uchar int vertex_indices\nend_header\n'
    s += ''.join(f'{x} {y} {z}\n' for x, y, z in TV)
    s += ''.join(f'3 {a} {b} {c}\n' for a, b, c in TF)
    return s


# current probe snapping variants
_ip = metal(box((10, 10, 3), (10, 10, 17)))
for n, (a, b) in enumerate([(8.3, 11.7), (8.7, 11.3), (8.5, 11.5), (8.2, 12.2), (7.8, 11.8), (8.49, 11.51), (8.51, 11.49)]):
    _ip += f'<ProbeBox Name="i{n}" Type="1" NormDir="2"><Primitives><Box Priority="0"><P1 X="{a}" Y="{a}" Z="12.2"/><P2 X="{b}" Y="{b}" Z="12.2"/></Box></Primitives></ProbeBox>'
C['probe_i_snap'] = model(_ip)

C['mat_aniso_single'] = model(mat(box((7, 7, 7), (13, 13, 13)), extra='Isotropy="0"').replace('Epsilon="4"', 'Epsilon="3" Kappa="0.01"'))
C['mat_aniso_two'] = model(mat(box((7, 7, 7), (13, 13, 13)), extra='Isotropy="0"').replace('Epsilon="4"', 'Epsilon="3,5" Kappa="0.01,0.02"'))
C['mat_iso_weight3'] = model(mat(box((5, 5, 5), (15, 15, 15)), eps='4', weight='Epsilon="1+0.1*x,1+0.1*y,1"'))
C['mat_aniso_weight1'] = model(mat(box((5, 5, 5), (15, 15, 15)), eps='2,3,4', extra='Isotropy="0"', weight='Epsilon="1+0.1*x"'))
C['tie_sphere_metal'] = model(metal(f'<Sphere Priority="0" Radius="4.5">{pt("Center", (10, 10, 10))}</Sphere>'))
C['tie_sphere_mat'] = model(mat(f'<Sphere Priority="0" Radius="4.5">{pt("Center", (10, 10, 10))}</Sphere>'))
C['tie_cyl_metal'] = model(metal(f'<Cylinder Priority="0" Radius="5">{pt("P1", (10, 10, 5))}{pt("P2", (10, 10, 15))}</Cylinder>'))
C['tie_cyl_mat'] = model(mat(f'<Cylinder Priority="0" Radius="2.5">{pt("P1", (10, 10, 5))}{pt("P2", (10, 10, 15))}</Cylinder>'))
C['tie_wire_z'] = model(metal(f'<Wire Priority="0" WireRadius="3">{pt("Vertex", (10, 10, 5))}{pt("Vertex", (10, 10, 15))}</Wire>'))
C['tie_wire_mat'] = model(mat(f'<Wire Priority="0" WireRadius="2.5">{pt("Vertex", (10, 10, 5))}{pt("Vertex", (10, 10, 15))}</Wire>'))
C['tie_cylshell_metal'] = model(metal(f'<CylindricalShell Priority="0" Radius="4" ShellWidth="2">{pt("P1", (10, 10, 4))}{pt("P2", (10, 10, 16))}</CylindricalShell>'))
C['tie_sphshell_mat'] = model(mat(f'<SphericalShell Priority="0" Radius="4" ShellWidth="1">{pt("Center", (10, 10, 10))}</SphericalShell>'))
C['tie_polygon_vertex'] = model(metal(f'<LinPoly Priority="0" NormDir="2" Elevation="8" Length="3" QtyVertices="4"><Vertex X1="6" X2="6"/><Vertex X1="14" X2="6"/><Vertex X1="14" X2="14"/><Vertex X1="6" X2="14"/></LinPoly>'))
C['tie_box_metal_cube'] = model(metal(f'<Polyhedron Priority="0">{CUBE_V}{CUBE_F}</Polyhedron>'))
C['exc_transformed'] = model('', source=False).replace('<Properties>', '''<Properties><Excitation Name="exc" Type="0" Excite="0,0,1">
        <Primitives><Box Priority="0"><P1 X="0" Y="0" Z="-2"/><P2 X="0" Y="0" Z="2"/><Transformation><Translate Argument="4,10,10"/></Transformation></Box></Primitives></Excitation>''')
C['dump_transformed'] = model('<DumpBox Name="Et" DumpType="0" DumpMode="0" FileType="1"><Primitives><Box Priority="0"><P1 X="-2" Y="-2" Z="0"/><P2 X="2" Y="2" Z="0"/><Transformation><Translate Argument="10,10,10"/></Transformation></Box></Primitives></DumpBox>', steps=40)
C['dump_two_prims'] = model('<DumpBox Name="E2" DumpType="0" DumpMode="0" FileType="1"><Primitives><Box Priority="0"><P1 X="8" Y="8" Z="10"/><P2 X="12" Y="12" Z="10"/></Box><Box Priority="0"><P1 X="8" Y="8" Z="5"/><P2 X="12" Y="12" Z="5"/></Box></Primitives></DumpBox>', steps=40)
C['probe_point_volt'] = model('<ProbeBox Name="up" Type="0"><Primitives><Point Priority="0" X="16" Y="6" Z="8"/></Primitives></ProbeBox>'
                              '<ProbeBox Name="hp" Type="3"><Primitives><Point Priority="0" X="12" Y="12" Z="10"/></Primitives></ProbeBox>')
C['probe_curve'] = model('<ProbeBox Name="uc" Type="0"><Primitives><Curve Priority="0"><Vertex X="16" Y="6" Z="8"/><Vertex X="16" Y="6" Z="12"/></Curve></Primitives></ProbeBox>')
C['probe_polygon_i'] = model(metal(box((10, 10, 3), (10, 10, 17))) + '<ProbeBox Name="ip" Type="1" NormDir="2"><Primitives><Polygon Priority="0" NormDir="2" Elevation="12"><Vertex X1="8" X2="8"/><Vertex X1="12" X2="8"/><Vertex X1="12" X2="12"/><Vertex X1="8" X2="12"/></Polygon></Primitives></ProbeBox>')
BIGX = ','.join(str(i) for i in range(41))
for n, bc in [('pml_big_8', 'xmin="PML_8" xmax="PML_8" ymin="PEC" ymax="PEC" zmin="PEC" zmax="PEC"'),
              ('pml_big_4', 'xmin="PML_4" xmax="PML_4" ymin="PEC" ymax="PEC" zmin="PEC" zmax="PEC"'),
              ('pml_big_3', 'xmin="3" xmax="PML_8" ymin="PEC" ymax="PEC" zmin="PEC" zmax="PEC"'),
              ('pml_big_plain', 'xmin="PML" xmax="PML_8" ymin="PEC" ymax="PEC" zmin="PEC" zmax="PEC"')]:
    C[n] = model('', bc=bc, grid=f'<RectilinearGrid DeltaUnit="1e-3" CoordSystem="0"><XLines>{BIGX}</XLines><YLines>{LINES}</YLines><ZLines>{LINES}</ZLines></RectilinearGrid>',
                 source=False, probes=False).replace('<Properties>', '''<Properties><Excitation Name="exc" Type="0" Excite="0,0,1">
        <Primitives><Box Priority="0"><P1 X="14" Y="10" Z="8"/><P2 X="14" Y="10" Z="12"/></Box></Primitives></Excitation>
      <ProbeBox Name="e1" Type="2"><Primitives><Box Priority="0"><P1 X="20" Y="10" Z="10"/><P2 X="20" Y="10" Z="10"/></Box></Primitives></ProbeBox>
      <ProbeBox Name="e2" Type="2"><Primitives><Box Priority="0"><P1 X="28" Y="6" Z="7"/><P2 X="28" Y="6" Z="7"/></Box></Primitives></ProbeBox>
      <ProbeBox Name="u1" Type="0"><Primitives><Box Priority="0"><P1 X="24" Y="6" Z="8"/><P2 X="24" Y="6" Z="12"/></Box></Primitives></ProbeBox>''')

_shapes = {
    'Sphere': f'<Sphere Priority="0" Radius="2">{pt("Center", (10, 10, 10))}</Sphere>',
    'Cylinder': f'<Cylinder Priority="0" Radius="2">{pt("P1", (10, 10, 8))}{pt("P2", (10, 10, 12))}</Cylinder>',
    'MultiBox': f'<MultiBox Priority="0">{pt("StartP", (8, 8, 10))}{pt("EndP", (12, 12, 10))}</MultiBox>',
    'Polygon': '<Polygon Priority="0" NormDir="2" Elevation="10"><Vertex X1="8" X2="8"/><Vertex X1="12" X2="8"/><Vertex X1="12" X2="12"/><Vertex X1="8" X2="12"/></Polygon>',
    'LinPoly': '<LinPoly Priority="0" NormDir="2" Elevation="9" Length="2"><Vertex X1="8" X2="8"/><Vertex X1="12" X2="8"/><Vertex X1="12" X2="12"/><Vertex X1="8" X2="12"/></LinPoly>',
    'Polyhedron': f'<Polyhedron Priority="0">{CUBE_V}{CUBE_F}</Polyhedron>',
    'Curve': f'<Curve Priority="0">{pt("Vertex", (8, 10, 10))}{pt("Vertex", (12, 10, 10))}</Curve>',
    'Wire': f'<Wire Priority="0" WireRadius="1">{pt("Vertex", (8, 10, 10))}{pt("Vertex", (12, 10, 10))}</Wire>',
    'Point': '<Point Priority="0" X="10" Y="10" Z="10"/>',
    'SphericalShell': f'<SphericalShell Priority="0" Radius="2" ShellWidth="1">{pt("Center", (10, 10, 10))}</SphericalShell>',
}
C['probe_shapes'] = model(''.join(
    f'<DumpBox Name="d{k}" DumpType="0" DumpMode="0" FileType="1"><Primitives>{v}</Primitives></DumpBox>'
    f'<ProbeBox Name="e{k}" Type="2"><Primitives>{v}</Primitives></ProbeBox>' for k, v in _shapes.items()), steps=20)

_ushapes = {
    'Box': box((16, 6, 8), (16, 6, 12)),
    'MultiBox': f'<MultiBox Priority="0">{pt("StartP", (16, 6, 8))}{pt("EndP", (16, 6, 12))}</MultiBox>',
    'Wire': f'<Wire Priority="0" WireRadius="0">{pt("Vertex", (16, 6, 8))}{pt("Vertex", (16, 6, 12))}</Wire>',
    'Curve': f'<Curve Priority="0">{pt("Vertex", (16, 6, 8))}{pt("Vertex", (16, 6, 12))}</Curve>',
    'Cylinder': f'<Cylinder Priority="0" Radius="0">{pt("P1", (16, 6, 8))}{pt("P2", (16, 6, 12))}</Cylinder>',
}
_ishapes = {
    'Box': box((8, 8, 12), (12, 12, 12)),
    'MultiBox': f'<MultiBox Priority="0">{pt("StartP", (8, 8, 12))}{pt("EndP", (12, 12, 12))}</MultiBox>',
    'Polygon': '<Polygon Priority="0" NormDir="2" Elevation="12"><Vertex X1="8" X2="8"/><Vertex X1="12" X2="8"/><Vertex X1="12" X2="12"/><Vertex X1="8" X2="12"/></Polygon>',
    'Tri': '<Polygon Priority="0" NormDir="2" Elevation="12"><Vertex X1="8" X2="8"/><Vertex X1="12" X2="8"/><Vertex X1="12" X2="12"/></Polygon>',
    'LinPoly0': '<LinPoly Priority="0" NormDir="2" Elevation="12" Length="0"><Vertex X1="8" X2="8"/><Vertex X1="12" X2="8"/><Vertex X1="12" X2="12"/><Vertex X1="8" X2="12"/></LinPoly>',
    'Cylinder': f'<Cylinder Priority="0" Radius="2">{pt("P1", (10, 10, 12))}{pt("P2", (10, 10, 12))}</Cylinder>',
    'Sphere0': f'<Sphere Priority="0" Radius="0">{pt("Center", (10, 10, 12))}</Sphere>',
}
C['probe_ui_shapes'] = model(metal(box((10, 10, 3), (10, 10, 17))) + ''.join(
    f'<ProbeBox Name="u{k}" Type="0"><Primitives>{v}</Primitives></ProbeBox>' for k, v in _ushapes.items()) + ''.join(
    f'<ProbeBox Name="i{k}" Type="1" NormDir="2"><Primitives>{v}</Primitives></ProbeBox>' for k, v in _ishapes.items()))

C['dump_optres2'] = model('<DumpBox Name="Eo2" DumpType="0" DumpMode="0" FileType="1" OptResolution="2,2.5,1"><Primitives><Box Priority="0"><P1 X="2" Y="2" Z="10"/><P2 X="18" Y="18" Z="12"/></Box></Primitives></DumpBox>'
                          '<DumpBox Name="Eo3" DumpType="0" DumpMode="2" FileType="1" OptResolution="4,3,0.5"><Primitives><Box Priority="0"><P1 X="2" Y="2" Z="10"/><P2 X="18" Y="18" Z="12"/></Box></Primitives></DumpBox>', steps=20)

C['exc_hard_h'] = model('', source=False).replace('<Properties>', '''<Properties><Excitation Name="exc" Type="3" Excite="0,1,0">
        <Primitives><Box Priority="0"><P1 X="4" Y="8" Z="9"/><P2 X="4" Y="12" Z="9"/></Box></Primitives></Excitation>''')
C['exc_soft_h_line'] = model('', source=False).replace('<Properties>', '''<Properties><Excitation Name="exc" Type="2" Excite="0,1,0">
        <Primitives><Box Priority="0"><P1 X="4" Y="8" Z="9"/><P2 X="4" Y="12" Z="9"/></Box></Primitives></Excitation>''')
C['exc_number'] = model('', source=True).replace('Name="exc" Type="0"', 'Name="exc" Type="0" Number="3"')
C['lorentz_weight'] = model('<LorentzMaterial Name="lor"><Property Epsilon="2" EpsilonPlasmaFrequency="5e9" EpsilonLorPoleFrequency="3e9" EpsilonRelaxTime="1e-9"/>'
                            '<Weight EpsilonPlasmaFrequency="1+0.02*x" Epsilon="1+0.02*y"/><Primitives>' + box((6, 6, 6), (14, 14, 14)) + '</Primitives></LorentzMaterial>', steps=800)
C['debye_weight'] = model('<DebyeMaterial Name="deb"><Property Epsilon="2" EpsilonDelta="3" EpsilonRelaxTime="3e-11"/>'
                          '<Weight EpsilonDelta="1+0.03*z"/><Primitives>' + box((6, 6, 6), (14, 14, 14)) + '</Primitives></DebyeMaterial>', steps=800)
C['drude_kappa'] = model('<LorentzMaterial Name="dru"><Property Epsilon="1.5" Kappa="0.02" EpsilonPlasmaFrequency="3e9" EpsilonRelaxTime="2e-10"/><Primitives>' + box((6, 6, 6), (14, 14, 14)) + '</Primitives></LorentzMaterial>', steps=800)
C['lumped_series_rlc'] = model('<LumpedElement Name="RLC" Direction="2" LEtype="1" R="20" L="2e-9" C="1e-12"><Primitives><Box Priority="0"><P1 X="14" Y="10" Z="9"/><P2 X="14" Y="10" Z="11"/></Box></Primitives></LumpedElement>')
C['sheet_thin_resistive'] = model('<ConductingSheet Name="cs" Conductivity="100" Thickness="1e-4"><Primitives><Box Priority="0"><P1 X="8" Y="2" Z="2"/><P2 X="8" Y="18" Z="18"/></Box></Primitives></ConductingSheet>')

X100 = ','.join(str(i) for i in range(101))
L2 = '0,1,2'


def line(props, bc='xmin="PEC" xmax="PEC" ymin="PMC" ymax="PMC" zmin="PEC" zmax="PEC"', steps=1500):
    return model(props, bc=bc, source=False, probes=False, steps=steps, ts='TimeStep="1.8e-12"',
                 grid=f'<RectilinearGrid DeltaUnit="1e-3" CoordSystem="0"><XLines>{X100}</XLines><YLines>{L2}</YLines><ZLines>{L2}</ZLines></RectilinearGrid>'
                 ).replace('<Properties>', '''<Properties><Excitation Name="exc" Type="0" Excite="0,0,1">
        <Primitives><Box Priority="0"><P1 X="10" Y="0" Z="0"/><P2 X="10" Y="2" Z="2"/></Box></Primitives></Excitation>
      <ProbeBox Name="ua" Type="0"><Primitives><Box Priority="0"><P1 X="40" Y="1" Z="0"/><P2 X="40" Y="1" Z="2"/></Box></Primitives></ProbeBox>
      <ProbeBox Name="ub" Type="0"><Primitives><Box Priority="0"><P1 X="75" Y="1" Z="0"/><P2 X="75" Y="1" Z="2"/></Box></Primitives></ProbeBox>
      <ProbeBox Name="uc" Type="0"><Primitives><Box Priority="0"><P1 X="90" Y="1" Z="0"/><P2 X="90" Y="1" Z="2"/></Box></Primitives></ProbeBox>''')


for sign in (0, 1):
    for t in (0, 1):
        C[f'abc_line_s{sign}_t{t}'] = line(f'<AbsorbingBC Name="abc" NormalSignPositive="{sign}" PhaseVelocity="299792458" AbsorbingBoundaryType="{t}"><Primitives><Box Priority="0"><P1 X="80" Y="0" Z="0"/><P2 X="80" Y="2" Z="2"/></Box></Primitives></AbsorbingBC>')
C['abc_line_ref_mur'] = line('', bc='xmin="PEC" xmax="MUR" ymin="PMC" ymax="PMC" zmin="PEC" zmax="PEC" MUR_PhaseVelocity="299792458"')
C['abc_line_ref_pec'] = line('')
G40 = ','.join(str(i) for i in range(41))
C['pml_open'] = model('', bc='xmin="PML_8" xmax="PML_8" ymin="PML_8" ymax="PML_8" zmin="PML_8" zmax="PML_8"',
                      grid=f'<RectilinearGrid DeltaUnit="1e-3" CoordSystem="0"><XLines>{G40}</XLines><YLines>{G40}</YLines><ZLines>{G40}</ZLines></RectilinearGrid>',
                      source=False, probes=False, steps=800).replace('<Properties>', '''<Properties><Excitation Name="exc" Type="0" Excite="0,0,1">
        <Primitives><Box Priority="0"><P1 X="20" Y="20" Z="18"/><P2 X="20" Y="20" Z="22"/></Box></Primitives></Excitation>
      <ProbeBox Name="e1" Type="2"><Primitives><Box Priority="0"><P1 X="26" Y="20" Z="20"/><P2 X="26" Y="20" Z="20"/></Box></Primitives></ProbeBox>
      <ProbeBox Name="e2" Type="2"><Primitives><Box Priority="0"><P1 X="28" Y="25" Z="24"/><P2 X="28" Y="25" Z="24"/></Box></Primitives></ProbeBox>
      <ProbeBox Name="e3" Type="2"><Primitives><Box Priority="0"><P1 X="15" Y="12" Z="13"/><P2 X="15" Y="12" Z="13"/></Box></Primitives></ProbeBox>
      <ProbeBox Name="u1" Type="0"><Primitives><Box Priority="0"><P1 X="24" Y="20" Z="18"/><P2 X="24" Y="20" Z="22"/></Box></Primitives></ProbeBox>''')

C['bc_spellings'] = model('', bc='xmin="PML" xmax="pmc" ymin="1.7" ymax="2.000000e+00" zmin="PML_0" zmax="Mur"')
C['fdtd_odd'] = model('', fdtd='OverSampling="0"', ts='TimeStepFactor="2"').replace('endCriteria="1e-30"', 'endCriteria="-1"')

_ps = ('<ParameterSet><Parameter Type="Const" name="a" value="0.1"/>'
       '<Parameter Type="Linear" name="w" value="3" min="1" max="5" step="1"/></ParameterSet>')
C['param_in_weight'] = model(mat(box((5, 5, 5), (15, 15, 15)), eps='4', weight='Epsilon="1+a*x"'), params=_ps)
C['param_linear'] = model(mat('<Box Priority="0"><P1 X="10-w" Y="7" Z="7"/><P2 X="10+w" Y="13" Z="13"/></Box>'), params=_ps)
C['param_in_exc_weight'] = model('', source=False, params=_ps).replace('<Properties>', '''<Properties><Excitation Name="exc" Type="0" Excite="0,0,1">
        <Weight X="0" Y="0" Z="1+a*y"/>
        <Primitives><Box Priority="0"><P1 X="4" Y="0" Z="8"/><P2 X="6" Y="20" Z="12"/></Box></Primitives></Excitation>''')
C['lumped_on_cyl'] = model('<LumpedElement Name="R" Direction="2" R="50"><Primitives><Cylinder Priority="0" Radius="1"><P1 X="14" Y="10" Z="9"/><P2 X="14" Y="10" Z="11"/></Cylinder></Primitives></LumpedElement>')
C['metal_two_prims_prio'] = model(metal(box((8, 8, 6), (12, 12, 14), 1)) + mat(box((9, 9, 9), (11, 11, 11), 5), eps='6'))

sel = sys.argv[2:]
for name, xml in C.items():
    if sel and name not in sel:
        continue
    d = os.path.join(S, 'cases', name)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, 'model.xml'), 'w').write(xml)
    if name == 'prim_stl':
        open(os.path.join(d, 'tet.stl'), 'w').write(stl())
    if name == 'prim_ply':
        open(os.path.join(d, 'tet.ply'), 'w').write(ply())
print(len(C), 'cases')
