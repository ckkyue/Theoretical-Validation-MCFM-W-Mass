!
!  SPDX-License-Identifier: GPL-3.0-or-later
!  Copyright (C) 2019-2022, respective authors of MCFM.
!
 
module nplotter_W
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

          allocate(histos(18))

          if (rank == 0) then
              write (*,*) "RESUMMATION: Using transition with switch ", transitionSwitch
          endif

        histos(1) = plot_setup_uniform(0.00_dp, 100.00_dp, 1.00_dp, 'pt34_fine')

        !   histos(1) = plot_setup_custom([0.0010d0,0.0013d0,0.0016d0,0.0020d0, &
        !         0.0025d0,0.0032d0,0.0040d0,0.0050d0,0.0063d0,0.0079d0, &
        !         0.0100d0,0.0126d0,0.0158d0,0.0200d0,0.0251d0,0.0316d0, &
        !         0.0398d0,0.0501d0,0.0631d0,0.0794d0,0.1000d0,0.1259d0, &
        !         0.1585d0,0.1995d0,0.2512d0,0.3162d0,0.3981d0,0.5012d0, &
        !         0.6310d0,0.7943d0,1.0000d0,1.2589d0,1.5849d0,1.9953d0, &
        !         2.5119d0,3.1623d0,3.9811d0,5.0119d0,6.3096d0,7.9433d0, &
        !         10.0000d0,12.5893d0,15.8489d0,19.9526d0,25.1189d0, &
        !         31.6228d0,39.8107d0,50.1187d0,63.0957d0,79.4328d0,100.0000d0], &
        !         'pt34_fine')

        histos(2) = plot_setup_custom([(0.25_dp*i, i=0, 14), 4.00_dp, 5.00_dp], 'y34')

        !   histos(2) = plot_setup_uniform(0.0_dp,60._dp,1.0_dp,'pt34')

        do i = 3, 18
            write(histo_name, '(a, i0)') 'pt34_', i
                histos(i) = plot_setup_uniform(0.00_dp, 100.00_dp, 1.00_dp, histo_name)
        end do

        !   histos(3) = plot_setup_custom([0d0,7.5d0,12.5d0,17.5d0, &
        !       24d0,30d0,40d0,50d0,70d0,110d0,150d0,190d0,250d0,600d0],'pt34_cms')

        !   histos(4) = plot_setup_custom([0d0,0.004d0,0.008d0,0.012d0, &
        !       0.016d0,0.02d0,0.024d0,0.029d0,0.034d0,0.039d0,0.045d0, &
        !       0.051d0,0.057d0,0.064d0,0.072d0,0.081d0,0.091d0,0.102d0, &
        !       0.114d0,0.128d0,0.145d0,0.165d0,0.189d0,0.219d0,0.258d0, &
        !       0.312d0,0.391d0,0.524d0,0.695d0,0.918d0,1.153d0,1.496d0, &
        !       1.947d0,2.522d0,3.277d0,5d0,10d0],'phistar_atlas')

        ! histos(2) = plot_setup_custom([-5.00_dp, (-4.00_dp+0.25_dp*i, i=0, 32), 5.00_dp], 'y34')

        ! do i = 3, 36
        !     write(histo_name, '(a, i0)') 'pt34_', i
        !         histos(i) = plot_setup_uniform(0.00_dp, 100.00_dp, 1.00_dp, histo_name)
        ! end do

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

          real(dp) :: pt34, trans
          real(dp) :: phistar, phiacop, costhetastar, delphi34

          real(dp) :: yrappure, y34
          real(dp) :: wt_array(18)
          integer :: i

          pt34 = pttwo(3,4,p)
        !   y34 = yrappure(p(3,:)+p(4,:))
          y34 = ABS(yrappure(p(3,:)+p(4,:)))
          delphi34 = delphi(p(3,:),p(4,:))
          phiacop = 2._dp*atan(sqrt((1._dp+cos(delphi34))/(1._dp-cos(delphi34))))
          costhetastar = tanh((etarap(3,p)-etarap(4,p))/2._dp)
          phistar = tan(phiacop/2._dp)*sin(acos(costhetastar))

          if (origKpart == kresummed) then
              if (abovecut .eqv. .false.) then
                  trans = transition((pt34/twomass(3,4,p))**2d0,0.001d0,transitionSwitch,0.001d0)
              else
                  ! fo piece without transition
                  trans = 1._dp
              endif
          else
              trans = 1._dp
          endif

          if (ieee_is_nan(pt34)) then
              pt34 = -1._dp
          endif

          if (ieee_is_nan(phistar)) then
              phistar = -1._dp
          endif

        ! slice 3 to 16
        do i = 3, 16
            if (y34 > 0.25*(i-3) .and. y34 < 0.25*(i-2)) then
                wt_array(i) = wt
            else
                wt_array(i) = 0._dp
            endif
        end do

        ! slice 17
        if (y34 > 3.5 .and. y34 < 4) then
            wt_array(17) = wt
        else
            wt_array(17) = 0._dp
        endif

        ! slice 18
        if (y34 > 4 .and. y34 < 5) then
            wt_array(18) = wt
        else
            wt_array(18) = 0._dp
        endif

        ! ! slice 3
        !   if (y34 > -5 .and. y34 < -4)then
        !       wt_array(3) = wt
        !   else
        !       wt_array(3) = 0._dp
        !   end if

        ! ! slice 36
        !   if (y34 > 4 .and. y34 < 5) then
        !       wt_array(36) = wt
        !   else
        !       wt_array(36) = 0._dp
        !   end if

        ! ! slice 4 to 35
        ! do i = 4, 35
        !     if (y34 > 0.25*(i-20) .and. y34 < 0.25*(i-19)) then
        !         wt_array(i) = wt
        !     else
        !         wt_array(i) = 0._dp
        !     endif
        ! end do

          ids = histos
          vals = [pt34, y34, (pt34, i=3, 18)]
          wts = [wt, wt, (wt_array(i), i=3, 18)]
        !   vals = [pt34, y34, (pt34, i=3, 36)]
        !   wts = [wt, wt, (wt_array(i), i=3, 36)]
        !   vals = [pt34,pt34,pt34,phistar]
        !   wts = [wt*trans,wt*trans,wt*trans,wt*trans]

      end subroutine

end module