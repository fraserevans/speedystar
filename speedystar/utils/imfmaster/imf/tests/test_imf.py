import pytest
import numpy as np
import itertools

from .. import imf

from ..imf import kroupa, chabrier2005


@pytest.mark.parametrize(('inp', 'out', 'rtol', 'atol'),
                         [(0.05, 5.6159, 1e-3, 1e-3),
                          (1.5, 0.0359, 1e-4, 1e-4),
                          (1.0, 0.0914, 1e-4, 1e-4),
                          (3.0, 0.0073, 1e-4, 1e-4),
                          (1, 0.0914, 1e-4, 1e-4),
                          (3, 0.0073, 1e-4, 1e-4)])
def test_kroupa_val(inp, out, rtol, atol):
    kroupa = imf.Kroupa()
    np.testing.assert_allclose(kroupa(inp), out, rtol=rtol, atol=atol)
    np.testing.assert_allclose(imf.kroupa(inp), out, rtol=rtol, atol=atol)


@pytest.mark.parametrize('massfunc', imf.massfunctions.keys())
def test_mmax(massfunc):
    """
    Regression test for issue #4
    """

    if (not hasattr(imf.get_massfunc(massfunc), 'mmin')):
        pytest.skip("{0} doesn't have mmin defined".format(massfunc))

    c = imf.make_cluster(10000, mmax=1, mmin=0.01, massfunc=massfunc)

    assert c.max() <= 1


@pytest.mark.parametrize(('mlow', 'mhigh'),
                         itertools.product((0.01, 0.08, 0.1, 0.5, 1.0, 0.03),
                                           (0.02, 0.08, 0.4, 0.5, 1.0, 120)))
def test_kroupa_integral(mlow, mhigh):
    if mlow >= mhigh:
        pytest.skip("mmin >= mmax")
    num = kroupa.integrate(mlow, mhigh, numerical=True)[0]
    anl = kroupa.integrate(mlow, mhigh, numerical=False)[0]

    np.testing.assert_almost_equal(num, anl)
    if num != 0:
        assert anl != 0


@pytest.mark.parametrize(('mlow', 'mhigh'),
                         itertools.product((0.01, 0.08, 0.1, 0.5, 1.0, 0.03),
                                           (0.02, 0.08, 0.4, 0.5, 1.0, 120)))
def test_kroupa_mintegral(mlow, mhigh):
    if mlow >= mhigh:
        pytest.skip("mmin >= mmax")
    num = kroupa.m_integrate(mlow, mhigh, numerical=True)[0]
    anl = kroupa.m_integrate(mlow, mhigh, numerical=False)[0]
    print("{0} {1} {2:0.3f} {3:0.3f}".format(mlow, mhigh, num, anl))
    np.testing.assert_almost_equal(num, anl)
    if num != 0:
        assert anl != 0


@pytest.mark.parametrize(('mlow', 'mhigh'),
                         itertools.product((0.033, 0.01, 0.08, 0.1, 0.5, 1.0, 0.03),
                                           (0.02, 0.05, 0.08, 0.4, 0.5, 1.0, 120)))
def test_chabrier_integral(mlow, mhigh):
    if mlow >= mhigh:
        pytest.skip("mmin >= mmax")

    num = chabrier2005.integrate(mlow, mhigh, numerical=True)[0]
    anl = chabrier2005.integrate(mlow, mhigh, numerical=False)[0]

    print("{0} {1} {2:0.3f} {3:0.3f}".format(mlow, mhigh, num, anl))
    np.testing.assert_almost_equal(num, anl)

    # for mlow in (0.01, 0.08, 0.1, 0.5, 1.0):
    #     for mhigh in (0.02, 0.08, 0.4, 0.5, 1.0):
    #         try:
    #             num = chabrier2005.m_integrate(mlow, mhigh, numerical=True)[0]
    #             anl = chabrier2005.m_integrate(mlow, mhigh, numerical=False)[0]
    #         except ValueError:
    #             continue
    #         print("{0} {1} {2:0.3f} {3:0.3f}".format(mlow, mhigh, num, anl))
    #         np.testing.assert_almost_equal(num, anl)


def test_make_cluster():
    cluster = imf.make_cluster(1000)
    assert np.abs(sum(cluster) - 1000 < 100)


def test_kroupa_inverses():
    assert np.abs(imf.inverse_imf(0, massfunc=imf.Kroupa(), mmin=0.01) - 0.01) < 2e-3
    assert np.abs(imf.inverse_imf(0, massfunc=imf.Kroupa(mmin=0.01)) - 0.01) < 2e-3
    assert np.abs(imf.inverse_imf(1, massfunc=imf.Kroupa(), mmax=200) - 200) < 1
    assert np.abs(imf.inverse_imf(1, massfunc=imf.Kroupa(mmax=200)) - 200) < 1


@pytest.mark.parametrize(('inp', 'out', 'rtol', 'atol'),
                         [(0.05, 5.6159, 1e-3, 1e-3),
                          (1.5, 0.0359, 1e-4, 1e-4),
                          (1.0, 0.0914, 1e-4, 1e-4),
                          (3.0, 0.0073, 1e-4, 1e-4),
                          (1, 0.0914, 1e-4, 1e-4),
                          (3, 0.0073, 1e-4, 1e-4)])
def test_kroupa_val_unchanged(inp, out, rtol, atol):
    # regression: make sure that imf.kroupa = imf.Kroupa
    kroupa = imf.Kroupa()
    np.testing.assert_allclose(kroupa(inp), out, rtol=rtol, atol=atol)
    np.testing.assert_allclose(imf.kroupa(inp), out, rtol=rtol, atol=atol)
    np.testing.assert_allclose(kroupa(inp), imf.kroupa(inp))
