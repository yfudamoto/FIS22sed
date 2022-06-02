### *FIS22*: Dust temperature and IR luminosity measurement method presented in Fudamoto, Inoue, and Sugahara 2022

Please feel free to contact me if you have any questions to [Yoshinobu Fudamoto](mailto:yoshinobu.fudamoto@gmail.com?subject=[GitHub]%20FIS22)

*__FIS22_free.py__*: This code calculate clumpiness parameter (Xi_clp), Td, Lir using multiple ALMA continuum measurements and dust continuum emission size.

- **Required inputs**:
    - Redshift (zS)
    - Gravitaional lensing magnification (= 1 if not magnified)
    - UV luminosity (Luv) and its error (eLuv) in erg/s
    - Dust continuum size (sarc) and its error (esarc) in erg/s
    - FIR observed wavelength (lobs) in micro meter
    - FIR observed fluxes (fnuobs) in mJy
    - error of FIR observed fluxes (efnuobs) in mJy

- **Some options you may edit**:
    - flg_plot, Do you want plots of best fit SEDs? yes: =1 no: =0
Plotting SEDs usually takes more time than not plotting

    - Number of iteration for the Monte-Carlo sampling (=3000-5000 in default) can be smaller if you just want to do tests
Larger values take more time than smaller values

    - Initial Guess for your dust mass (log Mdust/Msun) and clumpiness parameter (not log)

*__FIS22_fix.py__*: This is a code thta calcuate Td, Lir using single ALMA continuum measurements and dust continuum emission size.

- **Required inputs**:
    - Redshift (zS)
    - Gravitaional lensing magnification (= 1 if not magnified)
    - UV luminosity (Luv) and its error (eLuv) in erg/s
    - Dust continuum size (sarc) and its error (esarc) in erg/s
    - FIR observed wavelength (lobs) in micro meter
    - FIR observed fluxes (fnuobs) in mJy
    - error of FIR observed fluxes (efnuobs) in mJy

- **Some options you may edit**:
    - flg_plot, Do you want plots of best fit SEDs? yes: =1 no: =0
    Plotting SEDs usually takes more time than not plotting

    - Number of iteration for the Monte-Carlo sampling (=3000-5000 in default) can be smaller if you just want to do tests
    Larger values take more time than smaller values

    - Initial Guess for your dust mass (log Mdust/Msun)
    - Clumpiness parameter (log) and its error (log): default value is -1.02 $\pm$ 0.41

