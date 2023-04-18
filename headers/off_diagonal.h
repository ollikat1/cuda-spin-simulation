__global__ void off_diagonal_term(cuDoubleComplex *__restrict__ in1,
                                  cuDoubleComplex *__restrict__ in2,
                                  cuDoubleComplex *__restrict__ in3,
                                  cuDoubleComplex *__restrict__ in4,
                                  cuDoubleComplex *__restrict__ in5,
                                  const double *Mx, const double *My,
                                  const double t) {

  // Crank - Nicolson approximation of e^( - it H_3) * psi^(m)

  defaultIndexing;

  __shared__ cuDoubleComplex psi1[Nx], psi2[Nx], psi3[Nx], psi4[Nx], psi5[Nx],
      denominator[Nx], c12[Nx], c13[Nx], c23[Nx], c34[Nx], c35[Nx], c45[Nx];

  volatile unsigned const short idX = threadIdx.x;
  const cuDoubleComplex i = make_cuDoubleComplex(0.0, 1.0);

  if (idX < Nx) {

    psi1[idX] = in1[index];
    psi2[idX] = in2[index];
    psi3[idX] = in3[index];
    psi4[idX] = in4[index];
    psi5[idX] = in5[index];

    //    cuDoubleComplex fBxBy;

    // First, approximate psi^(m + 1/2) with forward Euler

    // Matrix elements of H_3[psi^(m)]

    // using denominator to store c_1 * F_ - (Bx - iBy)
    denominator[idX] = d_gs * (2.0 * (psi1[idX] * cuConj(psi2[idX]) +
                                      psi4[idX] * cuConj(psi5[idX])) +
                               sqrt(6.0) * (psi2[idX] * cuConj(psi3[idX]) +
                                            psi3[idX] * cuConj(psi4[idX]))) -
                       make_cuDoubleComplex(Mx[index], -My[index]);

    c12[idX] = denominator[idX] - 0.4 * d_ga * psi4[idX] * cuConj(psi5[idX]);
    c45[idX] = denominator[idX] - 0.4 * d_ga * psi1[idX] * cuConj(psi2[idX]);

    c13[idX] = 0.2 * d_ga * psi3[idX] * cuConj(psi5[idX]);
    c35[idX] = 0.2 * d_ga * psi1[idX] * cuConj(psi3[idX]);

    c23[idX] = sqrt(1.5) * denominator[idX] -
               0.2 * d_ga * psi3[idX] * cuConj(psi4[idX]);
    c34[idX] = sqrt(1.5) * denominator[idX] -
               0.2 * d_ga * psi2[idX] * cuConj(psi3[idX]);

    // psi^(m + 1/2) = (1 - it/2 * H_3) * psi^(m)
    /*
    psi1[idX] = psi1[idX] - i * t * 0.5 * (c12[idX] * in2[index] + c13[idX] *
    in3[index]); psi2[idX] = psi2[idX] - i * t * 0.5 * (cuConj(c12[idX]) *
    in1[index] + c23[idX] * in3[index]); psi3[idX] = psi3[idX] - i * t * 0.5 *
    (cuConj(c13[idX]) * in1[index] + c35[idX] * in5[index] + c34[idX] *
    in4[index] + cuConj(c23[idX]) * in2[index]); psi4[idX] = psi4[idX] - i * t *
    0.5 * (cuConj(c34[idX]) * in3[index] + c45[idX] * in5[index]); psi5[idX] =
    psi5[idX] - i * t * 0.5 * (cuConj(c35[idX]) * in3[index] + cuConj(c45[idX])
    * in4[index]);
    */
    // The above calculations reordered for slightly improved efficiency:

    psi1[idX] =
        psi1[idX] - i * t * 0.5 * (c12[idX] * psi2[idX] + c13[idX] * psi3[idX]);
    psi4[idX] =
        psi4[idX] -
        i * t * 0.5 * (cuConj(c34[idX]) * psi3[idX] + c45[idX] * psi5[idX]);

    psi2[idX] =
        psi2[idX] -
        i * t * 0.5 * (cuConj(c12[idX]) * in1[index] + c23[idX] * psi3[idX]);
    psi5[idX] = psi5[idX] - i * t * 0.5 *
                                (cuConj(c35[idX]) * psi3[idX] +
                                 cuConj(c45[idX]) * in4[index]);

    psi3[idX] =
        psi3[idX] - i * t * 0.5 *
                        (cuConj(c13[idX]) * in1[index] + c35[idX] * in5[index] +
                         c34[idX] * in4[index] + cuConj(c23[idX]) * in2[index]);

    // Elements of H_3 = H_3[psi^(m + 1/2)]

    // Storing c_1 * F_ - (Bx - iBy) in denominator again.
    denominator[idX] = d_gs * (2.0 * (psi1[idX] * cuConj(psi2[idX]) +
                                      psi4[idX] * cuConj(psi5[idX])) +
                               sqrt(6.0) * (psi2[idX] * cuConj(psi3[idX]) +
                                            psi3[idX] * cuConj(psi4[idX]))) -
                       make_cuDoubleComplex(Mx[index], -My[index]);

    c12[idX] = denominator[idX] - 0.4 * d_ga * psi4[idX] * cuConj(psi5[idX]);
    c45[idX] = denominator[idX] - 0.4 * d_ga * psi1[idX] * cuConj(psi2[idX]);

    c13[idX] = 0.2 * d_ga * psi3[idX] * cuConj(psi5[idX]);
    c35[idX] = 0.2 * d_ga * psi1[idX] * cuConj(psi3[idX]);

    c23[idX] = sqrt(1.5) * denominator[idX] -
               0.2 * d_ga * psi3[idX] * cuConj(psi4[idX]);
    c34[idX] = sqrt(1.5) * denominator[idX] -
               0.2 * d_ga * psi2[idX] * cuConj(psi3[idX]);

    // inverse(1 + it/2 * H_3[psi^(m + 1/2)) * psi^(m), matrix inversion from
    // mathematica

    denominator[idX] =
        32.0 * i +
        sqr(t) *
            (

                2.0 * i *
                    (

                        4.0 * (c34[idX] * cuConj(c34[idX]) +
                               c35[idX] * cuConj(c35[idX])) +
                        c23[idX] * cuConj(c23[idX]) *
                            (4.0 + c45[idX] * sqr(t) * cuConj(c45[idX]))

                            ) +
                (2.0 * i * c13[idX] + c12[idX] * c23[idX] * t) *
                    cuConj(c13[idX]) * (4.0 + sqr(cuConj(c45[idX])) * sqr(t)) +
                4.0 * (c34[idX] * c45[idX] * t * cuConj(c35[idX]) +
                       (2.0 * i * c45[idX] + c35[idX] * t * cuConj(c34[idX])) *
                           cuConj(c45[idX])) +
                cuConj(c12[idX]) *
                    (

                        c13[idX] * t * cuConj(c23[idX]) *
                            (4.0 + c45[idX] * sqr(t) * cuConj(c45[idX])) +
                        c12[idX] *
                            (

                                8.0 * i +
                                sqr(t) * (

                                             2.0 * i *
                                                 (c34[idX] * cuConj(c34[idX]) +
                                                  c35[idX] * cuConj(c35[idX]) +
                                                  c45[idX] * cuConj(c45[idX])) +

                                             t * (cuConj(c35[idX]) * c34[idX] *
                                                      c45[idX] +
                                                  c35[idX] * cuConj(c34[idX]) *
                                                      cuConj(c45[idX]))

                                                 ))));

    psi1[idX] =
        1.0 / denominator[idX] *
        (2.0 * i *
         (16.0 * in1[index] +
          t * (

                  2.0 * sqr(t) * i *
                      (c13[idX] * c34[idX] * c45[idX] * in5[index] +
                       c12[idX] * c23[idX] *
                           (c34[idX] * in4[index] + c35[idX] * in5[index])) +
                  c12[idX] * c23[idX] * c34[idX] * c45[idX] * in5[index] * t *
                      t * t -
                  8.0 * i * (c12[idX] * in2[index] + c13[idX] * in3[index]) -
                  4.0 *
                      (c12[idX] * c23[idX] * in3[index] +
                       c13[idX] * c34[idX] * in4[index] +
                       c13[idX] * c35[idX] * in5[index]) *
                      t

                  ) +

          (c23[idX] * in1[index] - c13[idX] * in2[index]) * sqr(t) *
              cuConj(c23[idX]) * (4.0 + c45[idX] * sqr(t) * cuConj(c45[idX])) -

          sqr(t) *
              (

                  (2.0 * i * c35[idX] + c34[idX] * c45[idX] * t) *
                      (2.0 * i * in1[index] + c12[idX] * in2[index] * t) *
                      cuConj(c35[idX]) +

                  (

                      -4.0 * c45[idX] * in1[index] +
                      2.0 * i *
                          (c12[idX] * c45[idX] * in2[index] +
                           c13[idX] * c45[idX] * in3[index] -
                           c13[idX] * c35[idX] * in4[index]) *
                          t +
                      c12[idX] * c23[idX] *
                          (c45[idX] * in3[index] - c35[idX] * in4[index]) *
                          sqr(t)

                          ) *
                      cuConj(c45[idX]) +

                  (2.0 * i * in1[index] + c12[idX] * in2[index] * t) *
                      cuConj(c34[idX]) *
                      (2.0 * i * c34[idX] + c35[idX] * t * cuConj(c45[idX]))

                      )

              ));

    // storing this here temporarily. No real performance diff, as the compiler
    // probably already does this. increases readability by a little bit, though.
    psi5[idX] = t * (2.0 * i * (c34[idX] * in4[index] + c35[idX] * in5[index]) +
                     c34[idX] * c45[idX] * in5[index] * t);

    psi2[idX] =
        1.0 / denominator[idX] *
        (2.0 * i *
         (16.0 * in2[index] +
          2.0 * i * c23[idX] * t * (-4.0 * in3[index] + psi5[idX]) -

          t * (cuConj(c12[idX]) *
                   (

                       8.0 * i * in1[index] -
                       c13[idX] * t * (-4.0 * in3[index] + psi5[idX]) +
                       sqr(t) * (

                                    in1[index] *
                                        (2.0 * i * c35[idX] +
                                         c34[idX] * c45[idX] * t) *
                                        cuConj(c35[idX]) +
                                    (2.0 * i * c45[idX] * in1[index] +
                                     c13[idX] * c45[idX] * in3[index] * t -
                                     c13[idX] * c35[idX] * in4[index] * t) *
                                        cuConj(c45[idX]) +
                                    in1[index] * cuConj(c34[idX]) *
                                        (2.0 * i * c34[idX] +
                                         c35[idX] * t * cuConj(c45[idX]))

                                        )

                           ) +
               t * (

                       (c23[idX] * in1[index] - c13[idX] * in2[index]) *
                           cuConj(c13[idX]) *
                           (4.0 + c45[idX] * sqr(t) * cuConj(c45[idX])) +

                       2.0 * i *
                           (

                               in2[index] * cuConj(c35[idX]) *
                                   (2.0 * i * c35[idX] +
                                    c34[idX] * c45[idX] * t) +
                               (2.0 * i * c45[idX] * in2[index] +
                                c23[idX] * c45[idX] * in3[index] * t -
                                c23[idX] * c35[idX] * in4[index] * t) *
                                   cuConj(c45[idX]) +
                               in2[index] * cuConj(c34[idX]) *
                                   (2.0 * i * c34[idX] +
                                    c35[idX] * t * cuConj(c45[idX]))

                                   )

                           )

                   )

              ));

    psi3[idX] =
        1.0 / denominator[idX] *
        (-2.0 * i *
         (

             -16.0 * in3[index] + 4.0 * psi5[idX] +
             t * (

                     4.0 * (-c45[idX] * in3[index] + c35[idX] * in4[index]) *
                         t * cuConj(c45[idX]) +
                     (2.0 * i * in1[index] + c12[idX] * in2[index] * t) *
                         cuConj(c13[idX]) *
                         (4.0 + c45[idX] * sqr(t) * cuConj(c45[idX])) +
                     2.0 * i * in2[index] * cuConj(c23[idX]) *
                         (4.0 + c45[idX] * sqr(t) * cuConj(c45[idX])) +

                     t * cuConj(c12[idX]) *
                         (in1[index] * cuConj(c23[idX]) *
                              (4.0 + c45[idX] * sqr(t) * cuConj(c45[idX])) +
                          c12[idX] * (-4.0 * in3[index] + psi5[idX] +
                                      (-c45[idX] * in3[index] +
                                       c35[idX] * in4[index]) *
                                          sqr(t) * cuConj(c45[idX]))

                              )

                         )

                 ));

    psi4[idX] =
        1.0 / denominator[idX] *
        (

            -2.0 * i *
            (-16.0 * in4[index] + 8.0 * i * c45[idX] * in5[index] * t +
             t * (

                     8.0 * i * in3[index] * cuConj(c34[idX]) +
                     t * (

                             4.0 * (c35[idX] * in5[index] * cuConj(c34[idX]) +
                                    (c45[idX] * in3[index] -
                                     c35[idX] * in4[index]) *
                                        cuConj(c35[idX])) +
                             cuConj(c23[idX]) *
                                 (

                                     -4.0 * c23[idX] * in4[index] +
                                     2.0 * i * c23[idX] * c45[idX] *
                                         in5[index] * t +
                                     4.0 * in2[index] * cuConj(c34[idX]) -
                                     2.0 * i * c45[idX] * in2[index] * t *
                                         cuConj(c35[idX])

                                         ) +
                             cuConj(c13[idX]) *
                                 (

                                     (2.0 * i * c13[idX] +
                                      c12[idX] * c23[idX] * t) *
                                         (2.0 * i * in4[index] +
                                          c45[idX] * in5[index] * t) +
                                     (2.0 * in1[index] -
                                      i * c12[idX] * in2[index] * t) *
                                         (2.0 * cuConj(c34[idX]) -
                                          i * c45[idX] * t * cuConj(c35[idX]))

                                         ) +
                             cuConj(c12[idX]) *
                                 (

                                     t * cuConj(c23[idX]) *
                                         (2.0 * i * c13[idX] * in4[index] +
                                          c13[idX] * c45[idX] * in5[index] * t -
                                          2.0 * i * in1[index] *
                                              cuConj(c34[idX]) -
                                          c45[idX] * in1[index] * t *
                                              cuConj(c35[idX])

                                              ) +
                                     c12[idX] *
                                         (

                                             -4.0 * in4[index] +
                                             2.0 * i * c45[idX] * in5[index] *
                                                 t +
                                             t *
                                                 (2.0 * i * in3[index] +
                                                  c35[idX] * in5[index] * t) *
                                                 cuConj(c34[idX]) +
                                             (c45[idX] * in3[index] -
                                              c35[idX] * in4[index]) *
                                                 sqr(t) * cuConj(c35[idX])

                                                 )

                                         )

                                 )

                         )

                 )

        );

    psi5[idX] =
        1.0 / denominator[idX] *
        (

            -2.0 * i *
            (

                -16.0 * in5[index] +
                t * (

                        2.0 * t * cuConj(c23[idX]) *
                            (

                                -2.0 * c23[idX] * in5[index] +
                                2.0 * in2[index] * cuConj(c35[idX]) +
                                i * t *
                                    (c23[idX] * in4[index] -
                                     in2[index] * cuConj(c34[idX])) *
                                    cuConj(c45[idX])

                                    ) +
                        t * cuConj(c13[idX]) *
                            (

                                -4.0 * c13[idX] * in5[index] +
                                2.0 * i * c12[idX] * c23[idX] * in5[index] * t +
                                (4.0 * in1[index] -
                                 2.0 * i * c12[idX] * in2[index] * t) *
                                    cuConj(c35[idX]) +
                                t *
                                    (

                                        2.0 * i * c13[idX] * in4[index] +
                                        c12[idX] * c23[idX] * in4[index] * t -
                                        2.0 * i * in1[index] *
                                            cuConj(c34[idX]) -
                                        c12[idX] * in2[index] * t *
                                            cuConj(c34[idX])

                                            ) *
                                    cuConj(c45[idX])

                                    ) +
                        4.0 * (

                                  (2.0 * i * in3[index] +
                                   c34[idX] * in4[index] * t) *
                                      cuConj(c35[idX]) +
                                  2.0 * i * in4[index] * cuConj(c45[idX]) +
                                  t * cuConj(c34[idX]) *
                                      (-1.0 * c34[idX] * in5[index] +
                                       in3[index] * cuConj(c45[idX]))

                                      ) +
                        t * cuConj(c12[idX]) *
                            (

                                t * cuConj(c23[idX]) *
                                    (2.0 * i * c13[idX] * in5[index] -
                                     2.0 * i * in1[index] * cuConj(c35[idX]) +
                                     t *
                                         (c13[idX] * in4[index] -
                                          in1[index] * cuConj(c34[idX])) *
                                         cuConj(c45[idX])

                                         ) +
                                c12[idX] *
                                    (-4.0 * in5[index] +
                                     t * (

                                             (2.0 * i * in3[index] +
                                              c34[idX] * in4[index] * t) *
                                                 cuConj(c35[idX]) +
                                             2.0 * i * in4[index] *
                                                 cuConj(c45[idX]) +
                                             t * cuConj(c34[idX]) *
                                                 (-1.0 * c34[idX] * in5[index] +
                                                  in3[index] * cuConj(c45[idX]))

                                                 ))

                                    )

                            )

                    )

        );

    // (1 - it/2 * H[psi^(m + 1/2)) * (intermediate psi)
    in1[index] =
        psi1[idX] - i * t * 0.5 * (c12[idX] * psi2[idX] + c13[idX] * psi3[idX]);
    in2[index] =
        psi2[idX] -
        i * t * 0.5 * (cuConj(c12[idX]) * psi1[idX] + c23[idX] * psi3[idX]);
    in3[index] =
        psi3[idX] - i * t * 0.5 *
                        (cuConj(c13[idX]) * psi1[idX] + c35[idX] * psi5[idX] +
                         c34[idX] * psi4[idX] + cuConj(c23[idX]) * psi2[idX]);
    in4[index] =
        psi4[idX] -
        i * t * 0.5 * (cuConj(c34[idX]) * psi3[idX] + c45[idX] * psi5[idX]);
    in5[index] = psi5[idX] - i * t * 0.5 *
                                 (cuConj(c35[idX]) * psi3[idX] +
                                  cuConj(c45[idX]) * psi4[idX]);
  }
}
