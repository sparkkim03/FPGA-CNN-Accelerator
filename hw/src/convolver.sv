`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Engineer: Stefano Park Kim 
// 
// Create Date: 07/01/2025 12:49:25 PM
// Design Name: Convolution Module 
// Module Name: convolver
// Project Name: FPGA_CNN_Accelerator 
// Target Devices: Realdigital AUP-ZU3 
// Description: 
// 
// 
// Revision:
// Revision 0.01 - File Created 07/01/2025 12:49:25 PM
// Revision 0.02 - Initial Write up of the module completed 07/01/2025 14:58:00 PM
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module convolver #(
    parameter integer n = 28,   // Input matrix size
    parameter integer k = 5,    // Kernel size
    parameter integer N = 16,   // Total bit width
    parameter integer Q = 12    // Fractional bit width
)(
    input logic clk_i,
    input logic rst_i,
    input logic en_i,
    input logic signed [N-1:0] activation_i,
    input logic [N-1:0] weights_i[0:k-1][0:k-1],
    output logic signed [N-1:0] conv_o,
    output logic val_conv_o,
    output logic done_conv_o
);
    // Internal signals
    // k - 1 rows must be buffered
    localparam NUM_LINE_BUFFER = k-1;

    logic [N-1:0] line_buffer [0:NUM_LINE_BUFFER-1][0:n-1];

    logic [N-1:0] window [0:k-1][0:k-1];

    logic [$clog2(n)-1:0] row_counter;
    logic [$clog2(n)-1:0] col_counter;
    
    //logic signed [N*2 - 1:0] conv_sum_temp;
    
    //assign conv_o  = conv_sum_temp >>> Q;

    //logic [$clog2((n-k+1)*(n-k+1)):0] output_counter; 

    //logic [N-2:0] conv_sum_temp;

    enum logic [1:0] {
        IDLE,
        PROCESSING,
        DONE
    } state, next_state;

    always_ff @(posedge clk_i) begin
        if (state == PROCESSING) begin
            for (int i = 0; i < k; i++) begin
                for (int j = k - 1; j > 0; j--) begin
                    window[i][j] <= window[i][j-1];
                end

                if (i == k - 1) begin
                    window[k-1][0] <= activation_i;
                end else begin
                    window[i][0] <= line_buffer[i][col_counter]; 
                end
            end

            if (NUM_LINE_BUFFER > 0) begin
                for (int i = 0; i < NUM_LINE_BUFFER; i++) begin
                    if (i == NUM_LINE_BUFFER - 1) begin
                        line_buffer[i][col_counter] <= activation_i; 
                    end else begin
                        line_buffer[i][col_counter] <= line_buffer[i+1][col_counter]; 
                    end
                end
            end
        end
    end

    // Calculate the output value
    always_comb begin
        conv_o = 0; // Initialize the sum
        for (int i = 0; i < k; i++) begin
            for (int j = 0; j < k; j++) begin
                // Adjust for the fractional bit width
                conv_o += ($signed(window[i][j]) * $signed(weights_i[i][j]));
            end
            //conv_o = conv_sum_temp;
        end
    end

    always_comb begin 
        done_conv_o = 0;
        case(state)
            IDLE: begin
                done_conv_o = 0;
            end
            PROCESSING: begin
                done_conv_o = 0;
            end
            DONE: begin
                done_conv_o = 1;
            end
        endcase
    end
    
    always_comb begin
        next_state = state; 
        case(state)
            IDLE: begin
                if(en_i) begin
                    next_state = PROCESSING;
                end
            end
            PROCESSING: begin
                if(row_counter == n-1 && col_counter == n-1) begin
                    next_state = DONE;
                end
            end
            DONE: begin
                if(!en_i) begin
                    next_state = IDLE;
                end
            end
            default: next_state = IDLE; 
        endcase
    end

    always_ff @(posedge clk_i or posedge rst_i) begin
        if(rst_i) begin
            // Async reset
            state <= IDLE;
            row_counter <= 0;
            col_counter <= 0;
            val_conv_o <= 0;
        end
        else begin
            state <= next_state;

            if(state == PROCESSING) begin
                if((row_counter >= k-1) && (col_counter >= k-1)) begin
                    val_conv_o <= 1;
                end
                else begin
                    val_conv_o <= 0;
                end
            
                if(col_counter == n-1) begin
                    col_counter <= 0;
                    row_counter <= row_counter+1;
                end
                else begin
                    col_counter <= col_counter+1;
                end
            end
            else if(state == IDLE && next_state == PROCESSING) begin
                row_counter <= 0;
                col_counter <= 0;
            end
            
            if(state == DONE) begin
                val_conv_o <= 0;
            end
        end 
    end
    
endmodule