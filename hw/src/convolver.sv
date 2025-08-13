`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Engineer: Stefano Park Kim 
// 
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

    logic [N-1:0] line_buffer [0:NUM_LINE_BUFFER-1][0:n-1];
    logic [N-1:0] weight_buffer;
    logic [N-1:0] feature_buffer;

    logic [N-1:0] window [0:k-1][0:k-1];
    logic [N-1:0] kernel [0:k-1][k-1:0];

    logic [$clog2(k)-1:0] kernel_x;
    logic [$clog2(k)-1:0] kernel_y;

    logic [$clog2(n)-1:0] row_counter;
    logic [$clog2(n)-1:0] col_counter;

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
            for (int i = 0; i < k; i++) begin
                for (int j = 0; j < k - 1; j++) begin
                    window[i][j] <= window[i][j+1];
                end

                if (i == k - 1) begin       
                    window[k-1][k-1] <= feature_buffer;
                end else begin
                    window[i][k-1] <= line_buffer[i][col_counter]; 
                end
            end

            if (NUM_LINE_BUFFER > 0) begin
                for (int i = 0; i < NUM_LINE_BUFFER; i++) begin
                    if (i == NUM_LINE_BUFFER - 1) begin
                        line_buffer[i][col_counter] <= feature_buffer; 
                    end else begin
                        line_buffer[i][col_counter] <= line_buffer[i+1][col_counter]; 
                    end
                end
            end
        end
    end

    always_ff @(posedge clk_i) begin
        weight_buffer <= weight_i;
        if(state == LOAD_KERNEL) begin
            kernel[kernel_y][kernel_x] <= weight_buffer;
        end
    end

    // Calculate the output value
    always_comb begin
        accum = bias_i; // Initialize the sum
        for (int i = 0; i < k; i++) begin
            for (int j = 0; j < k; j++) begin
                $display("for window[%d][%d](%d) * kernel[%d][%d](%d) = %d", i, j,window[i][j], i, j,kernel[i][j],$signed(window[i][j]) * $signed(kernel[i][j]));
                accum += ($signed(window[i][j]) * $signed(kernel[i][j]));
            end
        end
        $display("accum: %d", accum);
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
                if(kernel_x == k-1 && kernel_y == k-1) begin
                    next_state = PROCESSING;
                end
            end
            PROCESSING: begin
                if(row_counter == n-1 && col_counter == n-1) begin
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
            val_conv_o <= 0;
            kernel_x <= 0;
            kernel_y <= 0;
        end
        else begin
            case(state) 
                IDLE: begin
                    row_counter <= 0;
                    col_counter <= 0;
                    val_conv_o <= 0;
                    kernel_x <= 0;
                    kernel_y <= 0;
                end
                LOAD_KERNEL: begin
                    if(kernel_x == k - 1) begin
                        kernel_x <= 0;
                        kernel_y <= kernel_y + 1;
                    end
                    else begin
                        kernel_x <= kernel_x + 1;
                    end
                end
                PROCESSING: begin
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