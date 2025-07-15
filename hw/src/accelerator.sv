`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 07/15/2025 10:11:25 AM
// Design Name: 
// Module Name: accelerator
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module accelerator(
    input logic clk_i,
    input logic rst_i,
    input logic en_i,
    output logic [3:0] output_o
    );
    
    // ++++++++++++++++++++++++++++++++ INTERNAL CONSTANTS ++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    localparam N = 16;
    localparam Q = 12;
    
    localparam CONV1_SZ = 28;
    localparam CONV2_SZ = 12;
    localparam CONV_KERNEL = 5;
    
    localparam POOL1_SZ = 24;
    localparam POOL2_SZ = 8;
    localparam POOL_KERNEL = 2;
    
    localparam FC1_SZ = 256;
    localparam FC2_SZ = 128;
    localparam FC3_SZ = 84;
    localparam OUTPUT_SZ = 10;
    
    // ++++++++++++++++++++++++++++++++ INTERNAL SIGNALS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    // used to access the biases
    enum logic [2:0] {
        CONV1_BIAS,
        CONV2_BIAS,
        FC1_BIAS,
        FC2_BIAS,
        FC3_BIAS
    } bias_sel;
    
    // state machine
    enum logic [3:0] {
        IDLE,
        LOAD,
        CONV1,
        POOL1,
        CONV2,
        POOL2,
        DENSE1,
        DENSE2,
        DENSE3,
        DONE
    } state, next_state;
    
    logic [11:0] buffer_addr_read, buffer_addr_write;
    logic buffer_en_a, buffer_en_b, buffer_we;
    
    logic [N-1:0] window [0:CONV_KERNEL-1][0:CONV_KERNEL-1];
    
    // counts the number of outputs
    logic [10:0] counter_one;
    // counts the interation of output (such as which channel, which neuron)
    logic [6:0] counter_two;
    
    logic enable_conv1, enable_conv2, enable_pool1, enable_pool2, enable_fc1, enable_fc2, enable_fc3;
   
   // +++++++++++++++++++++++++++++++ BRAM Instatiation +++++++++++++++++++++++++++++++++++++++++++++++++++++
   
    data_buf buffer (
        .addra(buffer_addr_write),
        .clka(clk_i),
        .dina(),
        .ena(buffer_en_a),
        .wea(buffer_we),
        .addrb(buffer_addr_read),
        .doutb(),
        .enb(buffer_en_b)
    );
    
    conv1_weights conv1_weights (
        .addra(),
        .clka(),
        .dina('b0),
        .ena(),
        .wea(1'b0)
    );
    
    conv2_weights conv2_weights (
        .addra(),
        .clka(),
        .dina('b0),
        .ena(),
        .wea(1'b0)
    );
    
    fc1_weights fc1_weights (
        .addra(),
        .clka(),
        .dina('b0),
        .ena(),
        .wea(1'b0)
    );
    
    fc2_weights fc2_weights (
        .addra(),
        .clka(),
        .dina('b0),
        .ena(),
        .wea(1'b0)
    );
    
    fc3_weights fc3_weights (
        .addra(),
        .clka(),
        .dina('b0),
        .ena(),
        .wea(1'b0)
    );
    
    // ++++++++++++++++++++++++++++++++ MODULE Instantiation ++++++++++++++++++++++++++++++++++++++++++++++++
    
    // First Convolution layer
    convolver #(
        .n(CONV1_SZ),
        .k(CONV_KERNEL),
        .N(N),
        .Q(Q)
    ) conv_one (
        .clk_i(clk_i),
        .rst_i(rst_i),
        .en_i(enable_conv1),
        .activation_i(),
        .weights_i(),
        .bias_i(),
        .conv_o(),
        .val_conv_o(),
        .don_conv_o()
    );
    
    // First Pooling Layer
    max_pooler #(
        .n(POOL1_SZ),
        .s(POOL_KERNEL),
        .N(N)
    ) pooler_one (
        .clk_i(clk_i),
        .rst_i(rst_i),
        .en_i(enable_pool1),
        .data_i(),
        .data_o(),
        .val_pool_o(),
        .done_pool_()
    );
    
    // Second Convolution layer
    convolver #(
        .n(CONV2_SZ),
        .k(CONV_KERNEL),
        .N(N),
        .Q(Q)
    ) conv_two (
        .clk_i(clk_i),
        .rst_i(rst_i),
        .en_i(enable_conv2),
        .activation_i(),
        .weights_i(),
        .bias_i(),
        .conv_o(),
        .val_conv_o(),
        .don_conv_o()
    );
    
    // Second Pooling Layer
    max_pooler #(
        .n(POOL2_SZ),
        .s(POOL_KERNEL),
        .N(N)
    ) pooler_two (
        .clk_i(clk_i),
        .rst_i(rst_i),
        .en_i(enable_pool2),
        .data_i(),
        .data_o(),
        .val_pool_o(),
        .done_pool_()
    );
    
    // First dense layer
    dense #(
        .n(FC1_SZ),
        .m(FC2_SZ),
        .N(N),
        .Q(Q)
    ) fc_one (
        .clk_i(clk_i),
        .rst_i(rst_i),
        .en_i(enable_fc1),
        .data_i(),
        .bias_i(),
        .weight_i(),
        .weight_addr_o(),
        .data_o(),
        .val_dense_o(),
        .done_dense_o()
    );
    
    // Second dense layer
    dense #(
        .n(FC2_SZ),
        .m(FC3_SZ),
        .N(N),
        .Q(Q)
    ) fc_two (
        .clk_i(clk_i),
        .rst_i(rst_i),
        .en_i(enable_fc2),
        .data_i(),
        .bias_i(),
        .weight_i(),
        .weight_addr_o(),
        .data_o(),
        .val_dense_o(),
        .done_dense_o()
    );
    
    // Third dense layer
    dense #(
        .n(FC3_SZ),
        .m(OUTPUT_SZ),
        .N(N),
        .Q(Q)
    ) fc_three (
        .clk_i(clk_i),
        .rst_i(rst_i),
        .en_i(enable_fc3),
        .data_i(),
        .bias_i(),
        .weight_i(),
        .weight_addr_o(),
        .data_o(),
        .val_dense_o(),
        .done_dense_o()
    );
    
    bias_reg #(
        .N(N),
        .Q(Q)
    ) biases (
        .addr_i(),
        .sel_i(),
        .bias_o()
    );
    
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
endmodule
