`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Engineer: Stefano Park Kim 
// 
// Create Date: 07/01/2025 12:49:25 PM
// Design Name: Max Pooler Module 
// Module Name: max_pooler
// Project Name: FPGA_CNN_Accelerator 
// Target Devices: Realdigital AUP-ZU3 
// Description: modified the convolver module to perform pooling on the input
// 
// 
// Revision:
// Revision 0.01 - File Created 07/05/2025 11:32:25 PM
// Revision 0.02 - Initial Write up of the module completed 07/05/2025 14:58:00 PM
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module max_pooler # (
    parameter integer n = 24, // size of the input (n*n)
    parameter integer s = 2,  // size of the pooling window (s*s)
    parameter integer N = 16  // width of the data
)(
    input logic clk_i,
    input logic rst_i,
    input logic en_i,
    input logic signed [N-1:0] data_i,
    output logic signed [N-1:0] data_o,
    output logic val_pool_o,
    output logic done_pool_o
);
    
    logic [N-1:0] pool_buffer [0:n-1]; // lowkey just hard coded to s = 2 (only one row buffered)
    
    logic [N-1:0] pool_window [0:s-1][0:s-1]; 
    
    logic [$clog2(n)-1:0] row_counter;
    logic [$clog2(n)-1:0] col_counter;
    
    
    // counts the numbe
    logic [$clog2(n*n-1):0] input_counter;
    logic [$clog2((n/s)*(n/2)-1):0] output_counter;
    
    enum logic [1:0] {
        IDLE,
        PROCESSING,
        DONE
    } state, next_state;
    
     always_ff @(posedge clk_i) begin
        if (state == PROCESSING) begin
            for (int i = 0; i < s; i++) begin
                for (int j = s - 1; j > 0; j--) begin
                    pool_window[i][j] <= pool_window[i][j-1];
                end

                if (i == s - 1) begin
                    pool_window[s-1][0] <= data_i;
                end else begin
                    pool_window[i][0] <= pool_buffer[col_counter]; 
                end
            end

            input_counter <= input_counter + 1;
            pool_buffer[col_counter] <= data_i;
        end
    end

    logic [N-1:0] pool_temp;

    // determine the max value in the pooling window
    always_comb begin
        data_o = 0;
        // most negative number
        pool_temp = signed'({1'b1, {(N-1){1'b0}}});
        for(int i = 0; i < s; i++) begin
            for(int j = 0; j < s; j++) begin
                pool_temp = ($signed(pool_temp) > $signed(pool_window[i][j])) ? pool_temp : pool_window[i][j];
            end
        end
        data_o = (pool_temp > 0) ? pool_temp : 0;
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
                if(output_counter >= (n/s)*(n/s)) begin
                    next_state = DONE;
                end
            end
            DONE: begin
                if(!en_i) begin
                    next_state = IDLE;
                end
            end
        endcase
    end
    
    always_ff @(posedge clk_i) begin
        if(rst_i) begin
            state <= IDLE;
            row_counter <= 0;
            col_counter <= 0;
            input_counter <= 0;
            output_counter <= 0;
        end    
        else begin
            state <= next_state;
            
            if(state == PROCESSING) begin
                if(row_counter == s) begin
                    row_counter <= 0;
                    col_counter <= 0;
                    input_counter <= 1;
                end
                
                if((col_counter % s) - 1 == 0 && (input_counter+1 >= n + s)) begin
                    val_pool_o <= 1;
                    output_counter <= output_counter + 1;
                end
                else begin
                    val_pool_o <= 0;
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
        end
        
        if(state == DONE) begin
            val_pool_o <= 0;
        end;
    end
    
    
    
endmodule
