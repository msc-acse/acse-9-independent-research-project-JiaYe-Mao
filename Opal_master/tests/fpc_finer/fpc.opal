<?xml version='1.0' encoding='utf-8'?>
<OPAL>
  <opal_operation>
    <nirom>
      <snapshots_location>
        <path>
          <string_value lines="1">snapshots</string_value>
        </path>
        <input_file>
          <string_value lines="1">circleNRe3900_fixed.mpml</string_value>
        </input_file>
      </snapshots_location>
      <compression>
        <plain_autoencoder>
          <field_name name="Velocity">
            <batch_size>
              <integer_value rank="0">128</integer_value>
            </batch_size>
            <nsplt>
              <integer_value rank="0">1</integer_value>
            </nsplt>
            <generate_directory name="dd2_2_test"/>
            <dim>
              <integer_value rank="0">2</integer_value>
            </dim>
            <file_prefix name="Flowpast_2d_Re3900_"/>
            <number_epochs>
              <integer_value rank="0">1000</integer_value>
            </number_epochs>
            <number_latent_vector>
              <integer_value rank="0">32</integer_value>
            </number_latent_vector>
          </field_name>
        </plain_autoencoder>
      </compression>
      <training>
        <GPR>
          <scaling_bounds>
            <real_value shape="2" rank="1">0 10</real_value>
          </scaling_bounds>
          <constant_value>
            <real_value rank="0">1</real_value>
          </constant_value>
          <constant_bounds>
            <real_value shape="2" rank="1">1e-3 1e3</real_value>
          </constant_bounds>
          <RBF_length_scale>
            <real_value rank="0">100</real_value>
          </RBF_length_scale>
          <RBF_length_scale_bounds>
            <real_value shape="2" rank="1">1e-2 1e2</real_value>
          </RBF_length_scale_bounds>
        </GPR>
      </training>
    </nirom>
  </opal_operation>
</OPAL>
