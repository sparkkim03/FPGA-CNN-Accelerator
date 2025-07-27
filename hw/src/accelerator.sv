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
    input logic ld_i,
    input logic en_i,
    input logic [15:0] data_i,
    output logic [15:0] output_o [9:0]
    );
    
    // ++++++++++++++++++++++++++++++++ INTERNAL CONSTANTS ++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    localparam N = 16;
    localparam Q = 12;
    
    localparam INPUT_CH = 0;
        
    localparam CONV1_SZ = 28;
    localparam CONV2_SZ = 12;
    localparam CONV_KERNEL = 5;
    
    localparam CONV1_OUTPUT_CH = 6;
    localparam CONV2_OUTPUT_CH = 16;
    
    localparam POOL1_SZ = 24;
    localparam POOL2_SZ = 8;
    localparam POOL_KERNEL = 2;
    
    localparam FC1_SZ = 256;
    localparam FC2_SZ = 128;
    localparam FC3_SZ = 84;
    localparam OUTPUT_SZ = 10;
    
    localparam INPUT_SZ = CONV1_SZ * CONV1_SZ;
    
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
        LOADKERNEL1,
        POOL1,
        CONV2,
        LOADKERNEL2,
        POOL2,
        DENSE1,
        DENSE2,
        DENSE3,
        ACCUMULATE,
        DONE
    } state, next_state;
    
    logic [N-1:0] buffer_data_1, buffer_data_2;
    logic [11:0] buffer_addr_read_1, buffer_addr_write_1;
    logic [11:0] buffer_addr_read_2, buffer_addr_write_2;
    logic buffer_en_a_1, buffer_en_b_1, buffer_we_1;
    logic buffer_en_a_2, buffer_en_b_2, buffer_we_2;
    
    logic [7:0] conv1_w_addr;
    logic [11:0] conv2_w_addr;
    logic [14:0] fc1_w_addr;
    logic [13:0] fc2_w_addr;
    logic [9:0] fc3_w_addr;
    
    logic conv1_w_en, conv2_w_en, fc1_w_en, fc2_w_en, fc3_w_en;
    
    logic [N-1:0] conv1_w, conv2_w, conv_w;
    
    logic [N-1:0] kernel_window [0:CONV_KERNEL-1][0:CONV_KERNEL-1];
    
    
    logic [11:0] input_counter;
    // counts the interation of output (such as which channel, which neuron)
    // also counts input during the load phase
    logic [10:0] iter_counter;
    
    // coordinates for kernel
    logic [2:0] kernel_x, kernel_y;
    logic [2:0] kernel_x_dly, kernel_y_dly;
    
    // counts number of accumulation done
    logic [2:0] accum_counter;
    
    // accounts for the bram delay
    logic [1:0] startup_counter;
    
    logic enable_conv1, enable_conv2, enable_pool1, enable_pool2, enable_fc1, enable_fc2, enable_fc3;
    logic val_conv1, val_conv2, val_pool1, val_pool2, val_fc1, val_fc2, val_fc3;
    logic done_conv1, done_conv2, done_pool1, done_pool2, done_fc1, done_fc2, done_fc3;
    logic rst_conv1, rst_conv2;
    
    logic [N-1:0] conv1_in, pool1_in, conv2_in, pool2_in, fc1_in, fc2_in, fc3_in;
   
   // +++++++++++++++++++++++++++++++ BRAM Instatiation +++++++++++++++++++++++++++++++++++++++++++++++++++++
   
    data_buf buffer_1 (
        .addra(buffer_addr_write_1),
        .clka(clk_i),
        .dina(data_i),
        .ena(buffer_en_a_1),
        .wea(buffer_we_1),
        .addrb(buffer_addr_read_1),
        .doutb(buffer_data_1),
        .enb(buffer_en_b_1)
    );
    
    // it be useful to have a second buffer for the second convolution layer
    // since we have to convolve for 6, 16 channels
    data_buf buffer_2 (
        .addra(buffer_addr_write_2),
        .clka(clk_i),
        .dina(),
        .ena(buffer_en_a_1),
        .wea(buffer_we_1),
        .addrb(buffer_addr_read_2),
        .doutb(buffer_data_2),
        .enb(buffer_en_b_1)
    );
    
    conv1_weights conv1_weights (
        .addra(conv1_w_addr),
        .clka(clk_i),
        .dina('b0),
        .douta(conv1_w),
        .ena(conv1_w_en),
        .wea(1'b0)
    );
    
    conv2_weights conv2_weights (
        .addra(conv2_w_addr),
        .clka(clk_i),
        .dina('b0),
        .douta(conv2_w),
        .ena(conv2_w_en),
        .wea(1'b0)
    );
    
    fc1_weights fc1_weights (
        .addra(fc1_w_addr),
        .clka(clk_i),
        .dina('b0),
        .ena(fc1_w_en),
        .wea(1'b0)
    );
    
    fc2_weights fc2_weights (
        .addra(fc2_w_addr),
        .clka(clk_i),
        .dina('b0),
        .ena(fc2_w_en),
        .wea(1'b0)
    );
    
    fc3_weights fc3_weights (
        .addra(fc3_w_addr),
        .clka(clk_i),
        .dina('b0),
        .ena(fc3_w_en),
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
        .rst_i(rst_i | rst_conv1),
        .en_i(enable_conv1),
        .activation_i(conv1_in),
        .weights_i(kernel_window),
        .bias_i(),
        .conv_o(),
        .val_conv_o(val_conv1),
        .don_conv_o(done_conv1)
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
        .data_i(pool1_in),
        .data_o(),
        .val_pool_o(val_pool1),
        .done_pool_(done_pool1)
    );
    
    // Second Convolution layer
    convolver #(
        .n(CONV2_SZ),
        .k(CONV_KERNEL),
        .N(N),
        .Q(Q)
    ) conv_two (
        .clk_i(clk_i),
        .rst_i(rst_i | rst_conv2),
        .en_i(enable_conv2),
        .activation_i(conv2_in),
        .weights_i(kernel_window),
        .bias_i(),
        .conv_o(),
        .val_conv_o(val_conv2),
        .don_conv_o(done_conv2)
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
        .data_i(pool2_in),
        .data_o(),
        .val_pool_o(val_pool2),
        .done_pool_(done_pool2)
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
        .data_i(fc1_in),
        .bias_i(),
        .weight_i(),
        .weight_addr_o(),
        .data_o(),
        .val_dense_o(val_fc1),
        .done_dense_o(done_fc1)
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
        .data_i(fc2_in),
        .bias_i(),
        .weight_i(),
        .weight_addr_o(),
        .data_o(),
        .val_dense_o(val_fc2),
        .done_dense_o(done_fc2)
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
        .data_i(fc3_in),
        .bias_i(),
        .weight_i(),
        .weight_addr_o(),
        .data_o(),
        .val_dense_o(val_fc3),
        .done_dense_o(done_fc3)
    );
    
    bias_reg #(
        .N(N),
        .Q(Q)
    ) biases (
        .addr_i(),
        .sel_i(),
        .bias_o()
    );
    
    // ++++++++++++++++++++++++++++++++++++ DATA ROUTING +++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    always_comb begin
        buffer_addr_write_1 = 0;
        buffer_addr_write_2 = 0;
        
        buffer_addr_read_1 = 0;
        buffer_addr_read_2 = 0;
        
        case(state)
            LOAD: begin
                buffer_addr_write_1 = iter_counter;
            end
            CONV1: begin
                buffer_addr_read_1 = input_counter;
            end
        endcase
    end
    
    always_comb begin
        conv1_in = 0;
        conv2_in = 0;
        pool1_in = 0;
        pool2_in = 0;
        fc1_in = 0;
        fc2_in = 0;
        fc3_in = 0;
        
        unique case(state)
            CONV1: begin
                conv1_in = buffer_data_1;
            end
        endcase
    end
    
    // ++++++++++++++++++++++++++++++++++++ LOAD KERNEL +++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    always_comb begin
        if(state == LOADKERNEL1) begin
            conv1_w_addr = iter_counter * ('d25) + ('d5 * kernel_y) + kernel_x;
            conv_w = conv1_w;
        end
        else if(state == LOADKERNEL2) begin
            conv2_w_addr = iter_counter * ('d25) + ('d5 * kernel_y) + kernel_x;
            conv_w = conv2_w;
        end
        else begin
            conv1_w_addr = 0;
            conv2_w_addr = 0;
        end
    end
    
    always_ff @(posedge clk_i) begin
        if(state == LOADKERNEL1 || state == LOADKERNEL2) begin
            kernel_window[kernel_x_dly][kernel_y_dly] <= conv_w;
        end
    end
    
    // ++++++++++++++++++++++++++++++++++++ FSM LOGIC ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    always_comb begin
        buffer_en_a_1 = 0;
        buffer_we_1 = 0;
        
        buffer_en_a_2 = 0;
        buffer_we_2 = 0;
        
        rst_conv1 = 0;
        rst_conv2 = 0;
        
        enable_conv1 = 0;
        enable_conv2 = 0;
        
        conv1_w_en = 0;
    
        case(state)
            LOAD: begin
                buffer_addr_write_1 = iter_counter;
                buffer_en_a_1 = 1;
                buffer_we_1 = 1;
            end
            CONV1: begin
                if(startup_counter >= 2) begin
                    enable_conv1 = 1;
                end
            end
            LOADKERNEL1: begin
                rst_conv1 = 1;
                conv1_w_en = 1;
            end
        endcase
    end
    
    always_comb begin
        next_state = state;
        
        case(state)
            IDLE: begin
                if(ld_i) begin
                    next_state = LOAD;
                end
                
                if(en_i) begin
                    next_state = LOADKERNEL1;
                end
            end
            LOAD: begin
                if(iter_counter == INPUT_SZ - 1) begin
                    // counts 783
                    next_state = LOAD;
                end
            end
            CONV1: begin
                if(done_conv1) begin
                    if(iter_counter == CONV1_OUTPUT_CH * INPUT_CH - 1) begin
                        next_state = POOL1;
                    end
                    else begin
                        next_state = LOADKERNEL1;
                    end
                end
            end
            LOADKERNEL1: begin
                if(kernel_y == CONV_KERNEL - 1 && kernel_x == CONV_KERNEL - 1) begin
                    next_state = CONV1;
                end
            end
        endcase
    end
    
    always_ff @(posedge clk_i) begin
        if(state == IDLE) begin
            input_counter <= 0;
            iter_counter <= 0;
            kernel_x <= 0;
            kernel_y <= 0;
            startup_counter <= 0;
        end
        else if(state == LOAD) begin
            iter_counter <= iter_counter + 1;
        end
        else if(state == CONV1) begin
            kernel_x <= 0;
            kernel_y <= 0;
            
            input_counter <= input_counter + 1;
            
            if(startup_counter < 2) begin
                startup_counter <= startup_counter + 1;
            end
            
            if(done_conv1) begin
                iter_counter <= iter_counter + 1;
            end
        end
        else if(state == LOADKERNEL1) begin
            kernel_x <= kernel_x + 1;
            
            if(kernel_x == CONV_KERNEL - 1) begin
                kernel_x <= 0;
                kernel_y <= kernel_y + 1;
            end
            
            kernel_x_dly <= kernel_x;
            kernel_y_dly <= kernel_y;
        end
        
        if(state != next_state) begin
            if(next_state != LOADKERNEL1 || next_state != LOADKERNEL2) begin
                iter_counter <= 0;
                input_counter <= 0;
            end
            startup_counter <= 0;            
        end
    end
    
    always_ff @(posedge clk_i) begin
        if(rst_i) begin
            state <= IDLE;
        end
        else begin
            state <= next_state;
        end
    end
    
endmodule
