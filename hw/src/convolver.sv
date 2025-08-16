`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Engineer: Stefano Park Kim 
// 
// 
//////////////////////////////////////////////////////////////////////////////////


module convolver #(
    parameter integer n = 28,   // Input matrix size
    parameter integer k = 5,    // Kernel size
    parameter integer c = 6,    // Number of channels
    parameter integer N = 16   // Total bit width
)(
    input logic clk_i,
    input logic rst_i,
    input logic en_i,

    input logic signed [N-1:0] feature_i,

    input logic signed [N-1:0] weight_i,

    input logic signed [N-1:0] bias_i,

    output logic signed [N-1:0] conv_o,
    output logic val_conv_o,
    output logic done_conv_o
);
    // Internal signals
    // k - 1 rows must be buffered
    localparam NUM_LINE_BUFFER = k-1;

    logic [N-1:0] line_buffer [0:c-1][0:NUM_LINE_BUFFER-1][0:n-1];
    logic [N-1:0] weight_buffer;
    logic [N-1:0] feature_buffer;

    logic [N-1:0] window [0:c-1][0:k-1][0:k-1];
    logic [N-1:0] kernel [0:c-1][0:k-1][k-1:0];

    logic [$clog2(k)-1:0] kernel_x;
    logic [$clog2(k)-1:0] kernel_y;
    logic [$clog2(c)-1:0] kernel_c;

    logic [$clog2(n)-1:0] row_counter;
    logic [$clog2(n)-1:0] col_counter;
    logic [$clog2(c)-1:0] channel_counter;

    logic signed [2*N-1:0] accum;

    enum logic [1:0] {
        IDLE,
        LOAD_KERNEL,
        PROCESSING,
        DONE
    } state, next_state;

    always_ff @(posedge clk_i) begin
        feature_buffer <= feature_i;
        if (state == PROCESSING) begin
            // Update windows and line buffers for the current channel
            for (int i = 0; i < k; i++) begin
                for (int j = 0; j < k - 1; j++) begin
                    window[channel_counter][i][j] <= window[channel_counter][i][j+1];
                end

                if (i == k - 1) begin       
                    window[channel_counter][k-1][k-1] <= feature_buffer;
                end else begin
                    window[channel_counter][i][k-1] <= line_buffer[channel_counter][i][col_counter]; 
                end
            end

            if (NUM_LINE_BUFFER > 0) begin
                for (int i = 0; i < NUM_LINE_BUFFER; i++) begin
                    if (i == NUM_LINE_BUFFER - 1) begin
                        line_buffer[channel_counter][i][col_counter] <= feature_buffer; 
                    end else begin
                        line_buffer[channel_counter][i][col_counter] <= line_buffer[channel_counter][i+1][col_counter]; 
                    end
                end
            end
        end
    end

    always_ff @(posedge clk_i) begin
        weight_buffer <= weight_i;
        if(state == LOAD_KERNEL) begin
            kernel[kernel_c][kernel_y][kernel_x] <= weight_buffer;
        end
    end

    // Calculate the output value
    always_comb begin
        accum = bias_i; // Initialize the sum
        for (int ch = 0; ch < c; ch++) begin
            for (int i = 0; i < k; i++) begin
                for (int j = 0; j < k; j++) begin
                    //$display("for channel[%d] window[%d][%d](%d) * kernel[%d][%d](%d) = %d", ch, i, j, window[ch][i][j], i, j, kernel[ch][i][j], $signed(window[ch][i][j]) * $signed(kernel[ch][i][j]));
                    accum += ($signed(window[ch][i][j]) * $signed(kernel[ch][i][j]));
                end
            end
        end
        //$display("accum: %d", accum);
        conv_o = (accum > 0) ? accum : 0;
    end

    always_comb begin 
        done_conv_o = 0;
        
        if(state == DONE) done_conv_o = 1;
    end
    
    always_comb begin
        next_state = state; 
        case(state)
            IDLE: begin
                if(en_i) begin
                    next_state = LOAD_KERNEL;
                end
            end
            LOAD_KERNEL: begin
                if(kernel_x == k-1 && kernel_y == k-1 && kernel_c == c-1) begin
                    next_state = PROCESSING;
                end
            end
            PROCESSING: begin
                if(row_counter == n-1 && col_counter == n-1 && channel_counter == c-1) begin
                    next_state = DONE;
                end
            end
            DONE: begin
                next_state = IDLE;
            end
            default: next_state = IDLE; 
        endcase
    end
    
    always_ff @(posedge clk_i) begin
        if(rst_i) begin
            row_counter <= 0;
            col_counter <= 0;
            channel_counter <= 0;
            val_conv_o <= 0;
            kernel_x <= 0;
            kernel_y <= 0;
            kernel_c <= 0;
        end
        else begin
            case(state) 
                IDLE: begin
                    row_counter <= 0;
                    col_counter <= 0;
                    channel_counter <= 0;
                    val_conv_o <= 0;
                    kernel_x <= 0;
                    kernel_y <= 0;
                    kernel_c <= 0;
                end
                LOAD_KERNEL: begin
                    if(kernel_x == k - 1) begin
                        kernel_x <= 0;
                        if(kernel_y == k - 1) begin
                            kernel_y <= 0;
                            kernel_c <= kernel_c + 1;
                        end
                        else begin
                            kernel_y <= kernel_y + 1;
                        end
                    end
                    else begin
                        kernel_x <= kernel_x + 1;
                    end
                end
                PROCESSING: begin
                    if((row_counter >= k-1) && (col_counter >= k-1) && (channel_counter == c-1)) begin
                        val_conv_o <= 1;
                    end
                    else begin
                        val_conv_o <= 0;
                    end
                
                    if(channel_counter == c-1) begin
                        channel_counter <= 0;
                        if(col_counter == n-1) begin
                            col_counter <= 0;
                            row_counter <= row_counter+1;
                        end
                        else begin
                            col_counter <= col_counter+1;
                        end
                    end
                    else begin
                        channel_counter <= channel_counter + 1;
                    end
                end
            endcase
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