"�D
BHostIDLE"IDLE1�Q���@A�Q���@a�OK9��?i�OK9��?�Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1���Mb�}@9���Mb�}@A���Mb�}@I���Mb�}@ag�jt�?iZRT��?�Unknown
oHost_FusedMatMul"sequential/dense/Relu(1\���(�}@9\���(�}@A\���(�}@I\���(�}@a�5�9En�?i��Y�r)�?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1-����|@9-����|@A-����|@I-����|@afB� Fk�?i5�ro۶�?�Unknown
�HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(11�Zz@91�Zz@A1�Zz@I1�Zz@ag��Z@��?i��zc��?�Unknown
�HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1�/�$>S@9�/�$>S@A�/�$>S@I�/�$>S@a�>���?i���o��?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1�v��S@9�v��S@A�v��S@I�v��S@aC�߽�ڒ?i]~0G�?�Unknown
qHostCast"sequential/dropout/dropout/Cast(1J+�~P@9J+�~P@AJ+�~P@IJ+�~P@a�<P�J�?iBg�����?�Unknown
^	HostGatherV2"GatherV2(1���S�O@9���S�O@A���S�O@I���S�O@a<��/�?iw/!�Z�?�Unknown
�
HostDataset"4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat(1��x�&qG@9��x�&qG@Ay�&1�E@Iy�&1�E@a"�U�#H�?i8���{q�?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1T㥛�PD@9T㥛�PD@AT㥛�PD@IT㥛�PD@a�e��?i��"����?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1��ʡ�B@9��ʡ�B@A��ʡ�B@I��ʡ�B@aC�����?iLN�f�?�Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1�n��*B@9�n��*B@A�n��*B@I�n��*B@a�:[��?i����,T�?�Unknown
oHostSoftmax"sequential/dense_1/Softmax(1�C�l�k@@9�C�l�k@@A�C�l�k@@I�C�l�k@@a�@^@)8�?i�x����?�Unknown
oHostMul"sequential/dropout/dropout/Mul(1Zd;�O@@9Zd;�O@@AZd;�O@@IZd;�O@@a�fa��?iY;����?�Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1#��~j\:@9#��~j\:@A#��~j\:@I#��~j\:@am���Z	z?i�s��?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1��S�[9@9��S�[9@A��S�[9@I��S�[9@a��6�y?i�"
��:�?�Unknown�
iHostWriteSummary"WriteSummary(1R���0@9R���0@AR���0@IR���0@a!ߏ�]p?i��)]�[�?�Unknown�
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1J+��.@9J+��.@AJ+��.@IJ+��.@ay4� \un?i(�J�z�?�Unknown
rHostDataset"Iterator::Root::ParallelMapV2(1bX9�+@9bX9�+@AbX9�+@IbX9�+@ap����j?i�6�>Ɣ�?�Unknown
cHostDataset"Iterator::Root(1��n��:@9��n��:@A���Qx*@I���Qx*@a�cD�$j?i�H/)��?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1�|?5^�%@9�|?5^�%@A�|?5^�%@I�|?5^�%@air��e?i�%G���?�Unknown
�HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1R����"@9R����"@AR����"@IR����"@ao�����b?i�H��6��?�Unknown
qHostMul" sequential/dropout/dropout/Mul_1(1R���Q@9R���Q@AR���Q@IR���Q@aW�s�;�^?i�������?�Unknown
�HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1�x�&1�@9�x�&1�@A�x�&1�@I�x�&1�@a�#+]?i(�hD��?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1R���Q@9R���Q@AR���Q@IR���Q@a����\?i���پ�?�Unknown
�HostDataset">Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1��"���@9��"���@A��"���@I��"���@a&r��Y?i��(b��?�Unknown
�HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1㥛� �@9㥛� �@A㥛� �@I㥛� �@a�J:�/_Y?i�F�D�?�Unknown
[HostAddV2"Adam/add(1��~j�t@9��~j�t@A��~j�t@I��~j�t@a���S�'X?i���X)�?�Unknown
�HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1����K@9����K@A����K@I����K@a~�8�W?iq�mX5�?�Unknown
}HostMul",gradient_tape/sequential/dropout/dropout/Mul(1��n�@@9��n�@@A��n�@@I��n�@@a@�%,?�V?id���@�?�Unknown
 HostMul".gradient_tape/sequential/dropout/dropout/Mul_2(1;�O��n@9;�O��n@A;�O��n@I;�O��n@a>��'V?iޥ8�K�?�Unknown
~!HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1�MbX9@9�MbX9@A�MbX9@I�MbX9@aO��bQ�U?i���V�?�Unknown
e"Host
LogicalAnd"
LogicalAnd(1\���(\@9\���(\@A\���(\@I\���(\@a��87�U?i��na�?�Unknown�
�#HostDataset".Iterator::Root::ParallelMapV2::Zip[0]::FlatMap(1+�َ&@9+�َ&@A��Q�@I��Q�@a��py�R?i>T�j�?�Unknown
T$HostMul"Mul(1�S㥛�@9�S㥛�@A�S㥛�@I�S㥛�@a�XB�x�R?i<.�$t�?�Unknown
l%HostIteratorGetNext"IteratorGetNext(1;�O��n@9;�O��n@A;�O��n@I;�O��n@a�6F�7Q?i�9��|�?�Unknown
t&HostAssignAddVariableOp"AssignAddVariableOp(1�ʡE�s@9�ʡE�s@A�ʡE�s@I�ʡE�s@a�>�?P?i��z�߄�?�Unknown
`'HostGatherV2"
GatherV2_1(1X9��v@9X9��v@AX9��v@IX9��v@a&`�X�N?i:�P�e��?�Unknown
�(HostDataset"@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1R���Q@9R���Q�?AR���Q@IR���Q�?am�BQb�M?i�K� ��?�Unknown
\)HostArgMax"ArgMax_1(1��C�l@9��C�l@A��C�l@I��C�l@a{.���M?i��&��?�Unknown
w*HostDataset""Iterator::Root::ParallelMapV2::Zip(1�V�N@9�V�N@AP��n�@IP��n�@aR�bCb,K?i�Ω.��?�Unknown
v+HostAssignAddVariableOp"AssignAddVariableOp_2(1+����@9+����@A+����@I+����@a5w��"�H?i
�Zw��?�Unknown
[,HostPow"
Adam/Pow_1(1�rh��|@9�rh��|@A�rh��|@I�rh��|@a!ꘪ�/H?iE+�h'��?�Unknown
v-HostAssignAddVariableOp"AssignAddVariableOp_4(1u�V@9u�V@Au�V@Iu�V@a�7Ŕ��G?i�\�
��?�Unknown
�.HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1����K@9����K@A����K@I����K@a�X�C_G?i�I��ع�?�Unknown
Z/HostArgMax"ArgMax(1������@9������@A������@I������@a4-���F?i4H�y��?�Unknown
v0HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1{�G�z@9{�G�z@A{�G�z@I{�G�z@a'�y4F?i��!���?�Unknown
�1HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1y�&1�@9y�&1�@Ay�&1�@Iy�&1�@a��ƾgE?i����`��?�Unknown
V2HostSum"Sum_2(1��"��~@9��"��~@A��"��~@I��"��~@a�G�h>;E?i?�m����?�Unknown
v3HostAssignAddVariableOp"AssignAddVariableOp_1(1� �rh�@9� �rh�@A� �rh�@I� �rh�@a�i4��PD?iY������?�Unknown
w4HostReadVariableOp"div_no_nan/ReadVariableOp_1(1����S@9����S@A����S@I����S@a���%C?i����?�Unknown
X5HostEqual"Equal(1�������?9�������?A�������?I�������?a�\�i�k>?i�L(W��?�Unknown
�6HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1j�t��?9j�t��?Aj�t��?Ij�t��?a�<���=?i4����?�Unknown
�7HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1� �rh��?9� �rh��?A� �rh��?I� �rh��?aas��:;?i��s��?�Unknown
]8HostCast"Adam/Cast_1(1X9��v��?9X9��v��?AX9��v��?IX9��v��?a;]�J�y5?i���E"��?�Unknown
b9HostDivNoNan"div_no_nan_1(1�l�����?9�l�����?A�l�����?I�l�����?at�XF�0?i��R�9��?�Unknown
X:HostCast"Cast_2(1�S㥛��?9�S㥛��?A�S㥛��?I�S㥛��?a ��ŏ0?i�8�K��?�Unknown
�;HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1�Zd;�?9�Zd;�?A�Zd;�?I�Zd;�?a�W��.?i..�V9��?�Unknown
`<HostDivNoNan"
div_no_nan(17�A`���?97�A`���?A7�A`���?I7�A`���?a����o.?i�}�S ��?�Unknown
v=HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1���x�&�?9���x�&�?A���x�&�?I���x�&�?aV��k�,?iI7r���?�Unknown
�>HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1�p=
ף�?9�p=
ף�?A�p=
ף�?I�p=
ף�?a��`��I,?iW�����?�Unknown
t?HostReadVariableOp"Adam/Cast/ReadVariableOp(1�Q����?9�Q����?A�Q����?I�Q����?a6ٙ)?i3ד;K��?�Unknown
v@HostAssignAddVariableOp"AssignAddVariableOp_3(1���K7�?9���K7�?A���K7�?I���K7�?a挶���(?i�"����?�Unknown
�AHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1ˡE����?9ˡE����?AˡE����?IˡE����?a��D4�(?i�cR�d��?�Unknown
uBHostReadVariableOp"div_no_nan/ReadVariableOp(1)\���(�?9)\���(�?A)\���(�?I)\���(�?a�[o��'?iVY�y���?�Unknown
wCHostReadVariableOp"div_no_nan_1/ReadVariableOp(1V-���?9V-���?AV-���?IV-���?a��4�HS'?i��3�W��?�Unknown
oDHostReadVariableOp"Adam/ReadVariableOp(1;�O��n�?9;�O��n�?A;�O��n�?I;�O��n�?a>��'&?ipm�,���?�Unknown
yEHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1��ʡE��?9��ʡE��?A��ʡE��?I��ʡE��?a��I9)x#?iY����?�Unknown
YFHostPow"Adam/Pow(1��|?5^�?9��|?5^�?A��|?5^�?I��|?5^�?ab�a�T$"?i+X����?�Unknown
aGHostIdentity"Identity(1H�z�G�?9H�z�G�?AH�z�G�?IH�z�G�?a��i;!?ih�^%��?�Unknown�
vHHostCast"$sparse_categorical_crossentropy/Cast(1��ʡE��?9��ʡE��?A��ʡE��?I��ʡE��?a�"�^?i      �?�Unknown*�C
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1���Mb�}@9���Mb�}@A���Mb�}@I���Mb�}@a��W��T�?i��W��T�?�Unknown
oHost_FusedMatMul"sequential/dense/Relu(1\���(�}@9\���(�}@A\���(�}@I\���(�}@a�,�QP�?i8~
��R�?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1-����|@9-����|@A-����|@I-����|@aA������?i,�=� J�?�Unknown
�HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(11�Zz@91�Zz@A1�Zz@I1�Zz@a.4$Q�b�?i8��Y�b�?�Unknown
�HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1�/�$>S@9�/�$>S@A�/�$>S@I�/�$>S@a�E��4�?ie����S�?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1�v��S@9�v��S@A�v��S@I�v��S@a1߸8ߝ?i_pT�B�?�Unknown
qHostCast"sequential/dropout/dropout/Cast(1J+�~P@9J+�~P@AJ+�~P@IJ+�~P@a���A8ϙ?i��c��?�Unknown
^HostGatherV2"GatherV2(1���S�O@9���S�O@A���S�O@I���S�O@aRQQE6��?i?b�e���?�Unknown
�	HostDataset"4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat(1��x�&qG@9��x�&qG@Ay�&1�E@Iy�&1�E@aE��4�ې?i�i3W�]�?�Unknown
{
HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1T㥛�PD@9T㥛�PD@AT㥛�PD@IT㥛�PD@a�����ɏ?i$��
���?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1��ʡ�B@9��ʡ�B@A��ʡ�B@I��ʡ�B@aA�V޴��?i-���R�?�Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1�n��*B@9�n��*B@A�n��*B@I�n��*B@a�TtIm�?i�!���?�Unknown
oHostSoftmax"sequential/dense_1/Softmax(1�C�l�k@@9�C�l�k@@A�C�l�k@@I�C�l�k@@a�ݗ���?i��Qn+�?�Unknown
oHostMul"sequential/dropout/dropout/Mul(1Zd;�O@@9Zd;�O@@AZd;�O@@IZd;�O@@av��7�?it�¸J��?�Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1#��~j\:@9#��~j\:@A#��~j\:@I#��~j\:@a����Ο�?i��V����?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1��S�[9@9��S�[9@A��S�[9@I��S�[9@a��h��փ?i#k �%2�?�Unknown�
iHostWriteSummary"WriteSummary(1R���0@9R���0@AR���0@IR���0@a�a�?��y?i�G�� f�?�Unknown�
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1J+��.@9J+��.@AJ+��.@IJ+��.@aڵ�� x?iRY3�A��?�Unknown
rHostDataset"Iterator::Root::ParallelMapV2(1bX9�+@9bX9�+@AbX9�+@IbX9�+@a�����&u?i>m ���?�Unknown
cHostDataset"Iterator::Root(1��n��:@9��n��:@A���Qx*@I���Qx*@a� ;��t?i@�:���?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1�|?5^�%@9�|?5^�%@A�|?5^�%@I�|?5^�%@a�?�1q?ia�^�?�Unknown
�HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1R����"@9R����"@AR����"@IR����"@a4��rm?i��M��)�?�Unknown
qHostMul" sequential/dropout/dropout/Mul_1(1R���Q@9R���Q@AR���Q@IR���Q@aV��	�h?i�RB�?�Unknown
�HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1�x�&1�@9�x�&1�@A�x�&1�@I�x�&1�@a����g?i��4�lY�?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1R���Q@9R���Q@AR���Q@IR���Q@a��#v�f?i��?e]p�?�Unknown
�HostDataset">Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1��"���@9��"���@A��"���@I��"���@avC��Vd?i�����?�Unknown
�HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1㥛� �@9㥛� �@A㥛� �@I㥛� �@a�sSd?i�%A͘�?�Unknown
[HostAddV2"Adam/add(1��~j�t@9��~j�t@A��~j�t@I��~j�t@a�9)�A"c?i�N*^��?�Unknown
�HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1����K@9����K@A����K@I����K@a��36c?iD�A���?�Unknown
}HostMul",gradient_tape/sequential/dropout/dropout/Mul(1��n�@@9��n�@@A��n�@@I��n�@@a�e*1b?i��k�"��?�Unknown
HostMul".gradient_tape/sequential/dropout/dropout/Mul_2(1;�O��n@9;�O��n@A;�O��n@I;�O��n@a;��ߌa?i�}����?�Unknown
~ HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1�MbX9@9�MbX9@A�MbX9@I�MbX9@a���6ca?i�����?�Unknown
e!Host
LogicalAnd"
LogicalAnd(1\���(\@9\���(\@A\���(\@I\���(\@a�`��)�`?i1�����?�Unknown�
�"HostDataset".Iterator::Root::ParallelMapV2::Zip[0]::FlatMap(1+�َ&@9+�َ&@A��Q�@I��Q�@a��f	�]?i��u��?�Unknown
T#HostMul"Mul(1�S㥛�@9�S㥛�@A�S㥛�@I�S㥛�@a0�`5^]?iܣ:ym"�?�Unknown
l$HostIteratorGetNext"IteratorGetNext(1;�O��n@9;�O��n@A;�O��n@I;�O��n@a]�ri�F[?iB]o�0�?�Unknown
t%HostAssignAddVariableOp"AssignAddVariableOp(1�ʡE�s@9�ʡE�s@A�ʡE�s@I�ʡE�s@a�p�K�Y?i�>�<�?�Unknown
`&HostGatherV2"
GatherV2_1(1X9��v@9X9��v@AX9��v@IX9��v@aH��[��W?ih����H�?�Unknown
�'HostDataset"@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1R���Q@9R���Q�?AR���Q@IR���Q�?a�c뿸W?i5��8�T�?�Unknown
\(HostArgMax"ArgMax_1(1��C�l@9��C�l@A��C�l@I��C�l@a5`ІJW?ie�$�9`�?�Unknown
w)HostDataset""Iterator::Root::ParallelMapV2::Zip(1�V�N@9�V�N@AP��n�@IP��n�@a��1�V�U?i7�	�j�?�Unknown
v*HostAssignAddVariableOp"AssignAddVariableOp_2(1+����@9+����@A+����@I+����@a��"ΈS?i-`�p�t�?�Unknown
[+HostPow"
Adam/Pow_1(1�rh��|@9�rh��|@A�rh��|@I�rh��|@a����(S?i����U~�?�Unknown
v,HostAssignAddVariableOp"AssignAddVariableOp_4(1u�V@9u�V@Au�V@Iu�V@a0�B\$�R?i�*ؾ��?�Unknown
�-HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1����K@9����K@A����K@I����K@a���6�9R?i�[F�ې�?�Unknown
Z.HostArgMax"ArgMax(1������@9������@A������@I������@a��F_��Q?iU�uǙ�?�Unknown
v/HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1{�G�z@9{�G�z@A{�G�z@I{�G�z@a�g[�|�Q?i	��V���?�Unknown
�0HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1y�&1�@9y�&1�@Ay�&1�@Iy�&1�@a��p��P?i�����?�Unknown
V1HostSum"Sum_2(1��"��~@9��"��~@A��"��~@I��"��~@awp�<g�P?i8u*^u��?�Unknown
v2HostAssignAddVariableOp"AssignAddVariableOp_1(1� �rh�@9� �rh�@A� �rh�@I� �rh�@a~���P?i�Ӵ"���?�Unknown
w3HostReadVariableOp"div_no_nan/ReadVariableOp_1(1����S@9����S@A����S@I����S@a1��Z>N?i��q���?�Unknown
X4HostEqual"Equal(1�������?9�������?A�������?I�������?a��Da�H?i��I���?�Unknown
�5HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1j�t��?9j�t��?Aj�t��?Ij�t��?a��C��{G?i�;t����?�Unknown
�6HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1� �rh��?9� �rh��?A� �rh��?I� �rh��?aC�z6��E?i��MZ��?�Unknown
]7HostCast"Adam/Cast_1(1X9��v��?9X9��v��?AX9��v��?IX9��v��?a�(A?i
����?�Unknown
b8HostDivNoNan"div_no_nan_1(1�l�����?9�l�����?A�l�����?I�l�����?a���a�:?i���}���?�Unknown
X9HostCast"Cast_2(1�S㥛��?9�S㥛��?A�S㥛��?I�S㥛��?a��a��<:?i�3��?�Unknown
�:HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1�Zd;�?9�Zd;�?A�Zd;�?I�Zd;�?af�±io8?i9K<A��?�Unknown
`;HostDivNoNan"
div_no_nan(17�A`���?97�A`���?A7�A`���?I7�A`���?a+��8?iJ�4�D��?�Unknown
v<HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1���x�&�?9���x�&�?A���x�&�?I���x�&�?a@AH��6?iR�5d��?�Unknown
�=HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1�p=
ף�?9�p=
ף�?A�p=
ף�?I�p=
ף�?a����Ch6?iD��l���?�Unknown
t>HostReadVariableOp"Adam/Cast/ReadVariableOp(1�Q����?9�Q����?A�Q����?I�Q����?a�X~�zG4?i@\t��?�Unknown
v?HostAssignAddVariableOp"AssignAddVariableOp_3(1���K7�?9���K7�?A���K7�?I���K7�?a(cxz�3?iM����?�Unknown
�@HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1ˡE����?9ˡE����?AˡE����?IˡE����?a��S���3?i���^]��?�Unknown
uAHostReadVariableOp"div_no_nan/ReadVariableOp(1)\���(�?9)\���(�?A)\���(�?I)\���(�?ar����2?i|��=���?�Unknown
wBHostReadVariableOp"div_no_nan_1/ReadVariableOp(1V-���?9V-���?AV-���?IV-���?a�4��z2?icz~	��?�Unknown
oCHostReadVariableOp"Adam/ReadVariableOp(1;�O��n�?9;�O��n�?A;�O��n�?I;�O��n�?a;��ߌ1?if�l;��?�Unknown
yDHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1��ʡE��?9��ʡE��?A��ʡE��?I��ʡE��?aO�MI-�.?iBgA�(��?�Unknown
YEHostPow"Adam/Pow(1��|?5^�?9��|?5^�?A��|?5^�?I��|?5^�?a��ͽ,?i� z���?�Unknown
aFHostIdentity"Identity(1H�z�G�?9H�z�G�?AH�z�G�?IH�z�G�?a=!S'�	+?i������?�Unknown�
vGHostCast"$sparse_categorical_crossentropy/Cast(1��ʡE��?9��ʡE��?A��ʡE��?I��ʡE��?a�ʤ�e�%?i     �?�Unknown2CPU