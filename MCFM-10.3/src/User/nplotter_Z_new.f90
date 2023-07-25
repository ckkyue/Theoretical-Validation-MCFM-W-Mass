!
!  SPDX-License-Identifier: GPL-3.0-or-later
!  Copyright (C) 2019-2022, respective authors of MCFM.
!
 
module nplotter_Z
      use types
      use MCFMPlotting
      use ResummationTransition, only: transition
      use qtResummation_params, only: transitionSwitch
      implicit none

      public :: setup, book
      private

      integer, save, allocatable :: histos(:)

      contains

      subroutine setup()
          use types
          use parseinput
          implicit none
          integer :: i
          character(len=10) :: histo_name

          include 'mpicommon.f'

          include 'first.f'    
          if (first .and. rank == 0) then
            write(6,*) 'Using plotting routine nplotter_Z_new.f90'
            first=.false.
          endif
          allocate(histos(12))

          if (rank == 0) then
              write (*,*) "RESUMMATION: Using transition with switch ", transitionSwitch
          endif
        
        histos(1) = plot_setup_custom([0.0000d0,2.0000d0,3.0000d0,4.0000d0, &
                            5.0000d0,6.0000d0,7.0000d0,8.0000d0,9.0000d0, &
                            10.0000d0,12.0000d0,14.0000d0,16.0000d0,18.0000d0, &
                            20.0000d0,23.0000d0,27.0000d0,32.0000d0,40.0000d0, &
                            55.0000d0,100.0000d0], 'pt34_fine')

        histos(2) = plot_setup_uniform(0.00_dp,2.50_dp,0.25_dp,'y34')

        do i = 3, 12
            write(histo_name, '(a, i0)') 'pt34_', i
            histos(i) = plot_setup_custom([0.0000d0,2.0000d0,3.0000d0,4.0000d0, &
                            5.0000d0,6.0000d0,7.0000d0,8.0000d0,9.0000d0, &
                            10.0000d0,12.0000d0,14.0000d0,16.0000d0,18.0000d0, &
                            20.0000d0,23.0000d0,27.0000d0,32.0000d0,40.0000d0, &
                            55.0000d0,100.0000d0], histo_name)
        end do

          IF (.false.) THEN
            histos(1) = plot_setup_custom([0.0010d0,0.0013d0,0.0016d0,0.0020d0, &
                    0.0025d0,0.0032d0,0.0040d0,0.0050d0,0.0063d0,0.0079d0, &
                    0.0100d0,0.0126d0,0.0158d0,0.0200d0,0.0251d0,0.0316d0, &
                    0.0398d0,0.0501d0,0.0631d0,0.0794d0,0.1000d0,0.1259d0, &
                    0.1585d0,0.1995d0,0.2512d0,0.3162d0,0.3981d0,0.5012d0, &
                    0.6310d0,0.7943d0,1.0000d0,1.2589d0,1.5849d0,1.9953d0, &
                    2.5119d0,3.1623d0,3.9811d0,5.0119d0,6.3096d0,7.9433d0, &
                    10.0000d0,12.5893d0,15.8489d0,19.9526d0,25.1189d0, &
                    31.6228d0,39.8107d0,50.1187d0,63.0957d0,79.4328d0,100.0000d0], &
                    'pt34_fine')
            histos(2) = plot_setup_uniform(0.0_dp,60._dp,1.0_dp,'pt34')
            histos(3) = plot_setup_custom([0d0,2d0,4d0,6d0,8d0, &
                10d0,12d0,14d0,16d0,18d0,20d0,22.5d0,25d0,27.5d0,30d0, &
                33d0,36d0,39d0,42d0,45d0,48d0,51d0,54d0,57d0,61d0,65d0, &
                70d0,75d0,80d0,85d0,95d0,105d0,125d0,150d0,175d0,200d0, &
                250d0,300d0,350d0,400d0,470d0,550d0,650d0,900d0],'pt34_atlas')
            histos(4) = plot_setup_custom([0d0,0.004d0,0.008d0,0.012d0, &
                0.016d0,0.02d0,0.024d0,0.029d0,0.034d0,0.039d0,0.045d0, &
                0.051d0,0.057d0,0.064d0,0.072d0,0.081d0,0.091d0,0.102d0, &
                0.114d0,0.128d0,0.145d0,0.165d0,0.189d0,0.219d0,0.258d0, &
                0.312d0,0.391d0,0.524d0,0.695d0,0.918d0,1.153d0,1.496d0, &
                1.947d0,2.522d0,3.277d0,5d0,10d0],'phistar_atlas')

            histos(5) = plot_setup_custom([0d0,1d0,2d0,3d0,4d0,5d0,6d0,7d0, &
                8d0,9d0,10d0,11d0,12d0,13d0,14d0,16d0,18d0,20d0,22d0,25d0, &
                28d0,32d0,37d0,43d0,52d0,65d0,85d0,120d0,160d0,190d0,220d0, &
                250d0,300d0,400d0,500d0,800d0,1650d0],'pt34_cms13')

            histos(6) = plot_setup_custom([0d0,2.5d0,5d0,7.5d0,10d0,12.5d0, &
                15d0,17.5d0,20d0,30d0,40d0,50d0,70d0,90d0,110d0,150d0,190d0, &
                250d0, 600d0], 'pt34_cms7')
          END IF


      end subroutine

      subroutine book(p,wt,ids,vals,wts)
          use types
          use ResummationTransition
          use ieee_arithmetic
          implicit none
          include 'mxpart.f'
          include 'kpart.f'
          include 'taucut.f'! abovecut

          real(dp), intent(in) :: p(mxpart,4)
          real(dp), intent(in) :: wt

          integer, allocatable, intent(out) :: ids(:)
          real(dp), allocatable, intent(out) :: vals(:)
          real(dp), allocatable, intent(out) :: wts(:)

          real(dp) :: pttwo, twomass, delphi, etarap
          real(dp) :: pt34, trans04, trans06
          real(dp) :: phistar, phiacop, costhetastar, delphi34

          real(dp) :: yrappure, y34
          real(dp) :: wt_array(12) 
          integer :: i

          pt34 = pttwo(3,4,p)
          y34 = ABS(yrappure(p(3,:)+p(4,:)))
          delphi34 = delphi(p(3,:),p(4,:))
          phiacop = 2._dp*atan(sqrt((1._dp+cos(delphi34))/(1._dp-cos(delphi34))))
          costhetastar = tanh((etarap(3,p)-etarap(4,p))/2._dp)
          phistar = tan(phiacop/2._dp)*sin(acos(costhetastar))

          ! the variable transitionSwitch is taken from the input file and can be used here
          ! instead of the hardcoded 0.4 and 0.6

          if (origKpart == kresummed) then
              if (abovecut .eqv. .false.) then
                  trans04 = transition((pt34/twomass(3,4,p))**2d0,0.001d0, 0.4d0 ,0.001d0)
                  trans06 = transition((pt34/twomass(3,4,p))**2d0,0.001d0, 0.6d0 ,0.001d0)
              else
                  ! fo piece without transition
                  trans04 = 1._dp
                  trans06 = 1._dp
              endif
          else
              trans04 = 1._dp
              trans06 = 1._dp
          endif

          if (ieee_is_nan(pt34)) then
              pt34 = -1._dp
          endif

          if (ieee_is_nan(phistar)) then
              phistar = -1._dp
          endif

          ! fill histograms: first without transition function
          ! then with 0.4 transition function, then with 0.6 transition function
          ! for estimating matching uncertainty

        ! slice 3 to 12
        do i = 3, 12
            if (y34 > 0.25*(i-3) .and. y34 < 0.25*(i-2)) then
                wt_array(i) = wt
            else
                wt_array(i) = 0._dp
            endif
        end do

          ids = histos
          vals = [pt34, y34, pt34, pt34, pt34, pt34, pt34, pt34, pt34, pt34, pt34, pt34]
          wts = [wt, wt, wt_array(3), wt_array(4), wt_array(5), wt_array(6), wt_array(7), &
          wt_array(8), wt_array(9), wt_array(10), wt_array(11), wt_array(12)]

      end subroutine

end module