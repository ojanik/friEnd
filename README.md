# friEnd

**f**lux-**r**eady **i**nputs for the **E**nd-to-end optimization for **n**eutrino **d**etectors

A bunch of helpers to create input for [BESTE](https://github.com/ojanik/BESTIE/tree/main).

## Naming Conventions

To simplify working with different datasets (e.g., from different selections or even different experiments), a simple naming convention of datasets keys is introduced. To create a friEnd dataset this naming convetion is not required, however some helpers rely on this.

true_[Energy,Zenith,Azimuth,RA,Dec]: true MC 
reco_[Energy,Zenith,Azimuth,RA,Dec]_(method): reconstructed [energy, zenith, azimuth, right acension, declination] of the event. May be supplemented with a reconsturction method at the end, e.g., reco_Energy_transformer
fluxless_weight: weights that transform the injected spectrum to a powerlaw with norm=1 and index=0
